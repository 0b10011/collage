#![cfg_attr(feature = "external_doc", feature(external_doc))]
// This tests the README to ensure code snippets work.
#![cfg_attr(feature = "external_doc", doc(include = "../README.md"))]
//!

#![deny(missing_docs)]

extern crate rand;

use ansi_escapes::EraseLines;
use colored::*;
use image::{DynamicImage, FilterType, GenericImageView, ImageBuffer, ImageError, Rgb};
use indicatif::{ProgressBar, ProgressDrawTarget};
use log::{debug, info, log, Level};
use num_format::{Locale, ToFormattedString};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::cmp::{max, min};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process;
use std::sync::mpsc;
use std::u64;
use std::vec::Vec;
use threadpool::ThreadPool;

use std::error::Error;
use std::fmt;
use std::result;

/// All errors returned will be of this type
#[derive(Debug)]
pub enum CollageError {
    ///
    InternalError {
        ///
        description: String,
    },
    ///
    ImageError {
        ///
        description: String,
        ///
        source_error: ImageError,
    },
    ///
    IoError {
        ///
        description: String,
        ///
        source_error: std::io::Error,
    },
}

impl fmt::Display for CollageError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CollageError::InternalError { description } => write!(f, "{}", description),
            CollageError::ImageError {
                description,
                source_error: _,
            } => write!(f, "{}", description),
            CollageError::IoError {
                description,
                source_error: _,
            } => write!(f, "{}", description),
        }
    }
}

impl From<std::io::Error> for CollageError {
    fn from(err: std::io::Error) -> CollageError {
        CollageError::IoError {
            description: err.to_string(),
            source_error: err,
        }
    }
}

impl From<ImageError> for CollageError {
    fn from(err: ImageError) -> CollageError {
        CollageError::ImageError {
            description: err.to_string(),
            source_error: err,
        }
    }
}

impl Error for CollageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CollageError::InternalError { description: _ } => None,
            CollageError::ImageError {
                description: _,
                source_error,
            } => Some(source_error),
            CollageError::IoError {
                description: _,
                source_error,
            } => Some(source_error),
        }
    }
}

/// A result containing either an `ImageBuffer` or `CollageError`.
pub type CollageResult = result::Result<ImageBuffer<Rgb<u8>, Vec<u8>>, CollageError>;

type Result<T> = result::Result<T, CollageError>;

macro_rules! bad_file_error {
    ($skip_bad_files:expr, $($arg:tt)*) => (log!(if $skip_bad_files {Level::Warn} else {Level::Error}, $($arg)*))
}

struct RawImage {
    image: Result<DynamicImage>,
    x: u32,
    y: u32,
}

struct ImageInfo {
    old_width: u32,
    old_height: u32,
    new_width: u32,
    new_height: u32,
    crop_top: u32,
    crop_bottom: u32,
    crop_left: u32,
    crop_right: u32,
    path: Box<Path>,
}

struct CollageInfo {
    width: u32,
    height: u32,
    column_count: u32,
    column_width: u32,
    column_height_average: f64,
    columns: Vec<Column>,
    workers: usize,
    max_distortion: f32,
}

struct Column {
    height: u32,
    images: Vec<ImageInfo>,
}

/// Configuration for the collage.
pub struct CollageOptions<T>
where
    T: IntoIterator<Item = Box<Path>>,
{
    /// Width of the final collage in pixels.
    pub width: u32,
    /// Height of the final collage in pixels.
    pub height: u32,
    /// The image files to process and compile into a collage.
    pub files: T,
    /// Flag to skip invalid files instead of returning an error.
    pub skip_bad_files: bool,
    /// Number of workers for image processing.
    pub workers: usize,
    /// Max distortion of image height.
    ///
    /// If 0, images will be cropped to fit.
    /// Otherwise, after scaling proportionally to fit,
    /// the long dimension will be resized up to N% from it's proportional value.
    pub max_distortion: f32,
}

/// Generates the collage.
pub fn generate<I>(options: CollageOptions<I>) -> CollageResult
where
    I: IntoIterator<Item = Box<Path>>,
{
    if options.max_distortion > 100. {
        return Err(CollageError::InternalError {
            description: format!(
                "`max_distortion` cannot be greater than `100`. `{}` provided.",
                options.max_distortion
            ),
        });
    } else if options.max_distortion < 0. {
        return Err(CollageError::InternalError {
            description: format!(
                "`max_distortion` cannot be less than `0`. `{}` provided.",
                options.max_distortion
            ),
        });
    }

    let mut images: Vec<ImageInfo> =
        match get_images(options.files, options.skip_bad_files, options.workers) {
            Ok(images) => images,
            Err(err) => return Err(err),
        };

    info!(
        "Generating a {}x{} collage from {} images with {}% maximum distortion using {} workers.",
        options.width,
        options.height,
        images.len(),
        options.max_distortion,
        options.workers
    );

    normalize_images(&mut images);

    let mut collage_info = CollageInfo {
        width: options.width,
        height: options.height,
        column_count: 0,
        column_width: 0,
        column_height_average: 0.,
        columns: vec![],
        workers: options.workers,
        max_distortion: options.max_distortion,
    };

    add_columns(&mut collage_info, &mut images);

    while !balance_columns(&mut collage_info) && collage_info.column_count > 2 {
        remove_column(&mut collage_info);
    }

    create_collage(collage_info)
}

fn remove_column(collage_info: &mut CollageInfo) {
    collage_info.column_count -= 1;
    collage_info.column_width =
        (collage_info.width as f64 / collage_info.column_count as f64).ceil() as u32;
    let mut images = collage_info.columns.pop().unwrap().images;
    let mut column = collage_info.column_count as usize;
    while let Some(image) = images.pop() {
        column += 1;
        if column >= collage_info.column_count as usize {
            column = 0;
        }

        collage_info.columns[column].images.push(image);
    }

    for column in &mut collage_info.columns {
        column.height = 0;
        for image in &mut column.images {
            image.new_width = collage_info.column_width;
            image.new_height =
                (image.old_height as f64 / image.old_width as f64 * image.new_width as f64) as u32;
            column.height += image.new_height;
        }
    }

    info!(
        "Images should fit into {} columns that are {}px wide and average {}px tall.",
        collage_info.column_count.to_formatted_string(&Locale::en),
        collage_info.column_width.to_formatted_string(&Locale::en),
        (collage_info.column_height_average as i64).to_formatted_string(&Locale::en)
    );
}

fn create_collage(mut collage_info: CollageInfo) -> CollageResult {
    let mut collage = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(collage_info.width, collage_info.height);

    // Track how many columns are left in the canvas,
    // so we can distribute the offset between them.
    let mut remaining_count_x = collage_info.column_count;

    // Track how much we have to move the last column over
    // in order for it to appear aligned against the right edge.
    // We'll then distribute this amount between all of the columns,
    // so no one column is drastically cropped off the edge.
    let mut remaining_offset_x =
        ((collage_info.column_width * remaining_count_x) - collage_info.width) as i32;

    // The first column's offset should be 0.
    let mut x = 0_u32;

    let pool = ThreadPool::new(collage_info.workers);
    let (sender_base, receiver) = mpsc::channel();

    // Loop through columns and place their covers.
    remaining_count_x += 1; // Increment so we can decrement at start of loop
    let mut image_count = 0_u64;
    let mut removed_height = 0_u64;
    let mut total_height = 0_u64;
    let mut shortest_column = u64::MAX;
    let mut tallest_column = 1_u64;
    let mut rng = thread_rng();
    let mut alternate_column = false;
    for column in &mut collage_info.columns {
        // Decrement the remaining number of columns.
        remaining_count_x -= 1;

        column.images.shuffle(&mut rng);

        let mut short_key = 0;
        let mut short_height = column.height;
        let mut tall_key = 0;
        let mut tall_height = 0;
        for (key, image) in column.images.iter().enumerate() {
            if image.new_height >= tall_height {
                tall_key = key;
                tall_height = image.new_height;
            }
            if image.new_height <= short_height {
                short_key = key;
                short_height = image.new_height;
            }
        }

        // Move short image to first slot
        if short_key != 0 {
            column.images.swap(0, short_key);
        }
        // Move tall image to last slot
        let tall_image = column.images.swap_remove(tall_key);
        column.images.push(tall_image);
        // For every other column,
        // swap the short/tall images.
        if alternate_column {
            alternate_column = false;
            // Reverse images gets the intended results,
            // just with a bit more work.
            // Would be nice if this just swapped first/last images.
            column.images.reverse();
        } else {
            alternate_column = true;
        }

        // images.sort_unstable_by(|a, b| a.new_height.partial_cmp(&b.new_height).unwrap().reverse());

        // Crop from left and right of cover
        let remove_x: i32 =
            0 - (remaining_offset_x as f64 / remaining_count_x as f64).round() as i32;
        let crop_left = remove_x.abs() as u32 / 2;
        let crop_right = remove_x.abs() as u32 - crop_left;
        remaining_offset_x += remove_x;

        // Track how much we have to move the last cover up
        // in order for it to appear aligned against the bottom edge.
        // We'll then distribute this amount between all of the covers,
        // so no one cover is drastically cropped off the edge.
        let mut remaining_offset_y: i32 = column.height as i32 - collage_info.height as i32;

        // Track how much height is left in the column,
        // so we can distribute the offset between the covers accordingly.
        let mut remaining_height = column.height;

        // The first cover's offset should be 0.
        let mut y = 0;

        // Loop through covers in this column and place each one.
        while let Some(mut cover) = column.images.pop() {
            image_count += 1;

            // Crop cover based on amount needed to offset
            // proportional to the height of the image compared to the column.
            let remove_y: i32 = 0
                - (cover.new_height as f64 / remaining_height as f64 * remaining_offset_y as f64)
                    .round() as i32;
            remaining_height -= cover.new_height;
            cover.crop_top = remove_y.abs() as u32 / 2;
            cover.crop_bottom = remove_y.abs() as u32 - cover.crop_top;
            cover.crop_left = crop_left;
            cover.crop_right = crop_right;

            let cropped_width = (cover.new_width as i32 + remove_x) as u32;
            let cropped_height = (cover.new_height as i32 + remove_y) as u32;
            let resize_by_width = (cover.new_height as f64 / cover.new_width as f64
                * cropped_width as f64)
                .round() as u32
                >= cropped_height as u32;
            let old_width = cover.new_width;
            let old_height = cover.new_height;
            if resize_by_width {
                if remove_x != 0 {
                    cover.new_height = (cover.new_height as f64 / cover.new_width as f64
                        * cropped_width as f64)
                        .round() as u32;
                    cover.new_width = cropped_width as u32;
                }

                if cover.new_height != cropped_height {
                    cover.new_height = cropped_height.max(
                        (cover.new_height as f64 * (100.0 - collage_info.max_distortion as f64)
                            / 100.)
                            .round() as u32,
                    );
                }
            } else {
                if remove_y != 0 {
                    cover.new_width = (cover.new_width as f64 / cover.new_height as f64
                        * cropped_height as f64)
                        .round() as u32;
                    cover.new_height = cropped_height as u32;
                }

                if cover.new_width != cropped_width {
                    cover.new_width = cropped_width.max(
                        (cover.new_width as f64 * (100.0 - collage_info.max_distortion as f64)
                            / 100.)
                            .round() as u32,
                    );
                }
            }

            let adjustment_height = old_height - cover.new_height;
            let adjustment_top = adjustment_height / 2;
            cover.crop_top -= adjustment_top;
            cover.crop_bottom -= adjustment_height - adjustment_top;

            removed_height += adjustment_height as u64;

            let adjustment_width = old_width - cover.new_width;
            let adjustment_left = adjustment_width / 2;
            cover.crop_left -= adjustment_left;
            cover.crop_right -= adjustment_width - adjustment_left;

            // Adjust remaining y offset
            remaining_offset_y += remove_y;

            let adjusted_cover_height = cover.new_height - cover.crop_top - cover.crop_bottom;

            let sender = mpsc::Sender::clone(&sender_base);
            pool.execute(move || {
                sender
                    .send(RawImage {
                        image: get_resized_and_cropped_image(cover),
                        x: x,
                        y: y,
                    })
                    .unwrap();
                drop(sender);
            });

            // Set the starting y position for the next image
            y += adjusted_cover_height;
        }

        // Set offset for next column
        x += collage_info.column_width + remove_x as u32;
    }
    drop(sender_base);

    for column in &mut collage_info.columns {
        total_height += column.height as u64;
        shortest_column = min(column.height as u64, shortest_column);
        tallest_column = max(column.height as u64, tallest_column);
    }

    info!("{}px removed from {} columns with a sum height of {}px ({:.2}%) and split between {} covers (~{}px per cover).", removed_height.to_formatted_string(&Locale::en), collage_info.column_count.to_formatted_string(&Locale::en), total_height.to_formatted_string(&Locale::en), removed_height as f64 / total_height as f64 * 100., image_count.to_formatted_string(&Locale::en), ((removed_height as f64 / image_count as f64).round() as u64).to_formatted_string(&Locale::en));
    info!(
        "{}px removed from the shortest column with a height of {}px ({:.2}%).",
        (shortest_column as u64 - collage_info.height as u64).to_formatted_string(&Locale::en),
        shortest_column.to_formatted_string(&Locale::en),
        (shortest_column as u64 - collage_info.height as u64) as f64 / shortest_column as f64
            * 100.
    );
    info!(
        "{}px removed from the tallest column with a height of {}px ({:.2}%).",
        (tallest_column as u64 - collage_info.height as u64).to_formatted_string(&Locale::en),
        tallest_column.to_formatted_string(&Locale::en),
        (tallest_column as u64 - collage_info.height as u64) as f64 / tallest_column as f64 * 100.
    );

    info!("Resizing images and copying to collage...");
    let progress = ProgressBar::new(image_count);
    let (collage_width, collage_height) = collage.dimensions();
    while let Ok(raw_image) = receiver.recv() {
        progress.set_draw_target(ProgressDrawTarget::hidden());
        eprint!("{}", EraseLines(2));
        debug!("Placing image at {}x{}", raw_image.x, raw_image.y);
        progress.set_draw_target(ProgressDrawTarget::stderr());

        for pixel in raw_image.image?.to_rgb().enumerate_pixels() {
            if raw_image.x + pixel.0 >= collage_width {
                continue;
            }
            if raw_image.y + pixel.1 >= collage_height {
                continue;
            }

            collage.put_pixel(raw_image.x + pixel.0, raw_image.y + pixel.1, *pixel.2);
        }
        progress.inc(1);
    }
    progress.finish_and_clear();

    Ok(collage)
}

fn get_resized_and_cropped_image(cover: ImageInfo) -> Result<DynamicImage> {
    get_image(&cover.path).map(|image| {
        Ok(image
            .resize_exact(cover.new_width, cover.new_height, FilterType::Triangle)
            .crop(
                cover.crop_left,
                cover.crop_top,
                cover.new_width - cover.crop_left - cover.crop_right,
                cover.new_height - cover.crop_top - cover.crop_bottom,
            ))
    })?
}

fn balance_columns(collage_info: &mut CollageInfo) -> bool {
    // Continue looping back through until no images are swapped
    // to see if we can swap another pair.
    let mut swapped = true;
    let mut comparisons = 0;
    while swapped {
        swapped = false;

        // Sort columns from shortest to tallest
        collage_info
            .columns
            .sort_unstable_by(|a, b| a.height.partial_cmp(&b.height).unwrap());

        // Sort images by height in short and tall column
        collage_info
            .columns
            .first_mut()
            .unwrap()
            .images
            .sort_unstable_by(|a, b| a.new_height.partial_cmp(&b.new_height).unwrap().reverse());
        collage_info
            .columns
            .last_mut()
            .unwrap()
            .images
            .sort_unstable_by(|a, b| a.new_height.partial_cmp(&b.new_height).unwrap().reverse());

        let short_column = collage_info.columns.first().unwrap();
        let tall_column = collage_info.columns.last().unwrap();

        let difference_tall = (tall_column.height as f64 - collage_info.column_height_average)
            .abs()
            .floor() as u32;
        let difference_short = (collage_info.column_height_average - short_column.height as f64)
            .abs()
            .floor() as u32;

        let mut swap: [usize; 2] = [0, 0];
        for (tall_key, tall_image) in tall_column.images.iter().enumerate() {
            for (short_key, short_image) in short_column.images.iter().enumerate() {
                comparisons += 1;
                // If the difference between the tall column's cover's height
                // and the short column's cover's height
                // is greater than the difference between the column
                // closest to the average height
                // and the average height,
                // skip to the next cover.
                let difference_images = tall_image.new_height - short_image.new_height;
                if difference_tall >= difference_images || difference_short >= difference_images {
                    continue;
                }

                swap = [short_key, tall_key];

                // Track that we swapped covers
                // and break out of both foreach loops.
                swapped = true;
                break;
            }

            if swapped {
                break;
            }
        }

        if swapped {
            // Move short image from short column to tall column
            let short_image = collage_info
                .columns
                .first_mut()
                .unwrap()
                .images
                .swap_remove(swap[0]);
            collage_info.columns.first_mut().unwrap().height -= short_image.new_height;
            collage_info.columns.last_mut().unwrap().height += short_image.new_height;
            collage_info
                .columns
                .last_mut()
                .unwrap()
                .images
                .push(short_image);

            // Move tall image from tall column to short column
            let tall_image = collage_info
                .columns
                .last_mut()
                .unwrap()
                .images
                .swap_remove(swap[1]);
            collage_info.columns.last_mut().unwrap().height -= tall_image.new_height;
            collage_info.columns.first_mut().unwrap().height += tall_image.new_height;
            collage_info
                .columns
                .first_mut()
                .unwrap()
                .images
                .push(tall_image);
        } else if short_column.height < collage_info.height {
            info!(
                "Shortest column was {}px, but collage is supposed to be {}px tall.",
                short_column.height, collage_info.height
            );
            info!(
                "{} image height comparisons made for {} columns.",
                comparisons, collage_info.column_count
            );
            return false;
        }
    }

    info!(
        "{} image height comparisons made for {} columns.",
        comparisons, collage_info.column_count
    );
    true
}

fn add_columns(collage_info: &mut CollageInfo, images: &mut Vec<ImageInfo>) {
    // Sort from tallest to shortest
    // Without this,
    // we were seeing a 3.15%-3.98% height adjustment difference
    // degrade to a 1.67%-5.39% difference.
    // May have been due to test images used
    // (a couple hundred book covers and a few other images).
    images.sort_unstable_by(|a, b| a.new_height.partial_cmp(&b.new_height).unwrap());

    // Calculate max area
    let mut area: u64 = 0;
    for image in images.iter_mut() {
        area += (image.new_height * image.new_width) as u64;
    }

    // To figure out what size the images should be,
    // we do some math to calculate the width of column
    // based on the area:
    //
    //     width_adjustment * width * height_adjustment * height = image_area
    //
    // For the final collage,
    // both `width_adjustment` and `height_adjustment` should be `1`
    // (ie, no adjustment).
    // The `width` and `height` of the collage are provided,
    // so we can calculate the `image_area` from that.
    //
    // We also know the maximum area of the provided images
    // because we know how many pixels are in each image,
    // scaled to the same width as eachother.
    //
    // Since we're aiming to place the images
    // in a way that matches the aspect ratio of the final collage,
    // `width_adjustment` and `height_adjustment` should be the same number.
    // That simplifies the equation to:
    //
    //     adjustment * width * adjustment * height = image_area
    //
    // To solve for `adjustment`:
    //
    //     adjustment = sqrt(image_area / width / height)
    let adjustment =
        (area as f64 / collage_info.width as f64 / collage_info.height as f64).powf(0.5);
    let old_width = images[0].old_width;
    let mut new_width = (images[0].old_width as f64 / adjustment).ceil() as u32;
    collage_info.column_count = (collage_info.width as f64 / new_width as f64).floor() as u32;
    new_width = (collage_info.width as f64 / collage_info.column_count as f64).ceil() as u32;
    collage_info.column_width = new_width;

    // Add columns to collage_info
    for _ in 0..collage_info.column_count {
        collage_info.columns.push(Column {
            height: 0,
            images: vec![],
        });
    }

    // Calculate new size of images and average column height,
    // and add images to columns in order
    let mut column = collage_info.column_count as usize;
    let mut column_height_total: u64 = 0;
    while let Some(mut image) = images.pop() {
        column += 1;
        if column >= collage_info.column_count as usize {
            column = 0;
        }

        image.new_width = new_width;
        image.new_height = (image.old_height as f64 / old_width as f64 * new_width as f64) as u32;
        column_height_total += image.new_height as u64;
        collage_info.columns[column].height += image.new_height;
        collage_info.columns[column].images.push(image);
    }

    // Calculate average column height
    collage_info.column_height_average =
        column_height_total as f64 / collage_info.column_count as f64;

    info!(
        "Images will best fit into {} columns that are {}px wide and average {}px tall.",
        collage_info.column_count,
        collage_info.column_width,
        collage_info.column_height_average as i64
    );
}

fn normalize_images(sizes: &mut Vec<ImageInfo>) {
    let mut max_width: u32 = std::u32::MAX;
    let mut area: u64 = 0;
    for size in sizes.iter_mut() {
        max_width = min(max_width, size.old_width);
    }
    for size in sizes.iter_mut() {
        // Calculate scaled height (rounded)
        size.old_height =
            (max_width as f64 / size.old_width as f64 * size.old_height as f64).round() as u32;
        size.old_width = max_width;
        size.new_height = size.old_height;
        size.new_width = size.old_width;
        area += (size.new_height * size.new_width) as u64;
    }
    info!(
        "Narrowest image is {} pixels wide.",
        max_width.to_formatted_string(&Locale::en)
    );
    info!(
        "Maximum area of collage using all images is {} pixels.",
        area.to_formatted_string(&Locale::en)
    );
}

fn get_images<I>(files: I, skip_bad_files: bool, workers: usize) -> Result<Vec<ImageInfo>>
where
    I: IntoIterator<Item = Box<Path>>,
{
    let files = files.into_iter();
    let mut sizes: Vec<ImageInfo> = vec![];

    let pool = ThreadPool::new(workers);
    let (sender_base, receiver) = mpsc::channel();
    let mut file_count: u16 = 0;
    let mut succeeded_count: u16 = 0;

    info!("Retrieving image metadata.");
    for path in files {
        file_count += 1;
        let sender = mpsc::Sender::clone(&sender_base);
        pool.execute(move || {
            match get_image(&path) {
                Ok(image) => {
                    let (width, height) = image.dimensions();
                    sender
                        .send(Ok(ImageInfo {
                            old_width: width,
                            old_height: height,
                            new_width: width,
                            new_height: height,
                            crop_top: 0,
                            crop_bottom: 0,
                            crop_left: 0,
                            crop_right: 0,
                            path: path,
                        }))
                        .unwrap();
                }
                Err(err) => sender
                    .send(Err(format!(
                        "'{}' could not be added to the collage: {}",
                        path.to_string_lossy(),
                        err
                    )))
                    .unwrap(),
            }
            drop(sender);
        });
    }
    drop(sender_base);

    let bar = ProgressBar::new(file_count as u64);
    let mut exit_code = 0;
    while let Ok(result) = receiver.recv() {
        match result {
            Ok(image_info) => {
                bar.set_draw_target(ProgressDrawTarget::hidden());
                eprint!("{}", EraseLines(2));
                debug!("Valid file: {}", image_info.path.to_string_lossy());
                bar.set_draw_target(ProgressDrawTarget::stderr());

                succeeded_count += 1;
                sizes.push(image_info);
                bar.inc(1);
            }
            Err(err) => {
                bar.set_draw_target(ProgressDrawTarget::hidden());
                eprint!("{}", EraseLines(2));
                bad_file_error!(skip_bad_files, "Invalid file: {}", err);
                bar.set_draw_target(ProgressDrawTarget::stderr());

                exit_code = exitcode::DATAERR;
                bar.inc(1);
            }
        }
    }
    bar.finish_and_clear();

    if file_count == 0 {
        return Err(CollageError::InternalError {
            description: "No files provided.".to_string(),
        });
    }
    let failed_count = file_count - succeeded_count;
    if failed_count == 0 {
        info!(
            "{} images provided.",
            file_count.to_formatted_string(&Locale::en)
        );
    } else {
        bad_file_error!(
            skip_bad_files,
            "{}",
            format!("{} files were skipped.", failed_count)
                .on_red()
                .white()
        );
        bad_file_error!(
            skip_bad_files,
            "Only {} of {} files could be used.",
            succeeded_count,
            file_count
        );
    }

    if exit_code > 0 && !skip_bad_files {
        bad_file_error!(skip_bad_files,
            "Collage failed. One or more invalid files were provided. Use '--skip-bad-files' to igore those files."
        );
        process::exit(exit_code);
    }

    Ok(sizes)
}

fn get_image(path: &Path) -> Result<DynamicImage> {
    let mut buffer = Vec::new();
    let mut file = File::open(path)?;
    file.read_to_end(&mut buffer)?;
    Ok(image::load_from_memory(&buffer)?)
}
