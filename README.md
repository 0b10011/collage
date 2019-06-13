# Collage [![Build status](https://travis-ci.org/0b10011/collage.svg?branch=master)](https://travis-ci.org/0b10011/collage)

Generates a collage from a collection of images,
which is a great way to visually represent tens or hundreds of images or photographs.

## Library (Rust)

```rust
use collage::{CollageOptions, CollageResult, generate};
use std::{env, fs};
use std::path::Path;

let mut files: Vec<Box<Path>> = vec![];
// Add files to vector here

let collage: CollageResult = collage::generate(CollageOptions {
    width: 500,
    height: 500,
    files: files,
    skip_bad_files: false,
    workers: num_cpus::get(),
    max_distortion: 3.0,
});
```

## Binary (CLI)

```text
USAGE:
    collage [FLAGS] [OPTIONS] --height <height> --width <width>

FLAGS:
        --help              Prints help information
    -q, --quiet             Shows less detail. -q shows less detail, -qq shows least detail, -qqq is equivalent to
                            --silent.
    -s, --silent            Silences all output.
        --skip-bad-files    Ignore files in the wrong format or that don't exist.
    -V, --version           Prints version information
    -v, --verbose           Shows more detail. -v shows more detail, -vv shows most detail.

OPTIONS:
    -h, --height <height>                    Sets the final height of the collage.
        --max_distortion <max_distortion>    Max distortion of height or width. If 0, images will be cropped to fit.
                                             Otherwise, after scaling proportionally to fit, the long dimension will be
                                             resized up to N% from it's proportional value.
    -w, --width <width>                      Sets the final width of the collage.
        --workers <workers>                  Number of workers for image processing. Defaults to number of CPUs.
```

## How it works

1. Images will be measured and dimensions scaled to the narrowest width.
2. Number of columns will be estimated from provided width and height, and the size of the images
3. Images will be split into columns
4. Images will be swapped between columns to get all of the columns to a similar height:
   1. Each column will be measured.
   2. If an image from the shortest and tallest columns can be swapped to get closer to the average height, do so and repeat these steps.
   3. Otherwise, move on to the next step.
5. Randomize order of images within a column.
6. Avoid top/bottom edges of images from lining up with neighboring columns:
   1. Move shortest image to top and tallest image to bottom in odd columns.
   2. Move tallest image to top and shortest image to bottom in even columns.
7. Resize all images from their full size to fit in a column.
8. Calculate the extra height in each column
9. Split that extra height among images proportionally in each column, and for each image:
   1. Shorten up to `max_distortion` (eg, if an image is 100px tall and `max_distortion` is set to `5`, the final image would be no shorter than 95px).
   2. Crop the remaining amount from the top/bottom.
10. Place images in collage based on column and position within column.
