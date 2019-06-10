use clap::{crate_version, value_t, App, Arg};
use log::{info, LevelFilter};
use num_format::{Locale, ToFormattedString};
use std::path::Path;
use std::time::Instant;
use std::vec::Vec;
use std::{env, fs, u64};
use collage::{generate, CollageOptions};

fn main() {
    let now = Instant::now();

    let options = App::new("Collage Generator")
        .version(crate_version!())
        .author("Brandon Frohs <brandon@19.codes>")
        .about("Generates a collage from a collection of images.")
        // Dimensions
        .arg(
            Arg::with_name("width")
                .short("w")
                .long("width")
                .help("Sets the final width of the collage.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("height")
                .short("h")
                .long("height")
                .help("Sets the final height of the collage.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("workers")
                .long("workers")
                .help("Number of workers for image processing. Defaults to number of CPUs.")
                .takes_value(true),
        )
        // Ignore bad files
        .arg(
            Arg::with_name("skip-bad-files")
                .long("skip-bad-files")
                .help("Ignore files in the wrong format or that don't exist."),
        )
        // Verbosity level
        .arg(
            Arg::with_name("silent")
                .short("s")
                .long("silent")
                .conflicts_with_all(&["quiet", "verbose"])
                .help("Silences all output."),
        )
        .arg(
            Arg::with_name("quiet")
                .short("q")
                .multiple(true)
                .long("quiet")
                .conflicts_with("verbose")
                .help("Shows less detail. -q shows less detail, -qq shows least detail, -qqq is equivalent to --silent."),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .multiple(true)
                .long("verbose")
                .help("Shows more detail. -v shows more detail, -vv shows most detail."),
        )
        .get_matches();

    // Default log level is Info
    // --silent switches level to Off
    // -v, --verbose increases level to debug
    // -vv increases level to Trace
    // -q, --quiet decreases level to Error
    // -qq decreases level to Off
    let filter: LevelFilter = if options.is_present("silent") {
        LevelFilter::Off
    } else {
        match options.occurrences_of("verbose") {
            0 => match options.occurrences_of("quiet") {
                3...u64::MAX => LevelFilter::Off,
                2 => LevelFilter::Error,
                1 => LevelFilter::Warn,
                0 => LevelFilter::Info, // Default
            },
            1 => LevelFilter::Debug,
            2...u64::MAX => LevelFilter::Trace,
        }
    };

    env_logger::Builder::from_default_env()
        .filter(Some(module_path!()), filter)
        .init();

    let dir = env::current_dir().unwrap().join("src/images");
    let mut files: Vec<Box<Path>> = vec![];
    for dir in fs::read_dir(dir).expect("Directory doesn't seem to exist") {
        files.push(dir.unwrap().path().into_boxed_path());
    }

    let collage = generate(CollageOptions {
        width: value_t!(options.value_of("width"), u32).unwrap_or_else(|e| e.exit()),
        height: value_t!(options.value_of("height"), u32).unwrap_or_else(|e| e.exit()),
        files: files,
        skip_bad_files: options.is_present("skip-bad-files"),
        workers: value_t!(options.value_of("workers"), usize).unwrap_or(num_cpus::get()),
    });

    collage.unwrap().save("collage.png").unwrap();

    info!("Saved to 'collage.png'.");

    info!(
        "Generated in {}.{}s",
        now.elapsed().as_secs().to_formatted_string(&Locale::en),
        now.elapsed().subsec_millis()
    );
}
