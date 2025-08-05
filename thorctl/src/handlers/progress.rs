//! Progress bar utilities

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use owo_colors::OwoColorize;
use std::{borrow::Cow, time::Duration};

use crate::errors::Errors;

/// The different types of progress bars
pub enum BarKind {
    /// A bar with only a simple timer
    Timer,
    /// An unbounded bar with a spinner
    Unbound,
    /// A bounded bar
    Bound(u64),
    /// An unbounded IO bar
    UnboundIO,
    /// An IO bar
    #[allow(dead_code)]
    IO,
}

impl BarKind {
    /// The speed at which to spin the spinner
    const SPINNER_INTERVAL_MILLIS: u64 = 100;

    /// Configure this bar to the right type
    ///
    /// # Arguments
    ///
    /// * `name` - The name for this bar
    pub fn setup(self, name: &str, bar: &ProgressBar) {
        // configure this bar to the right kind
        match self {
            BarKind::Timer => {
                // build our style string
                let style = format!("[{{elapsed_precise}}] {{spinner}} {name} {{msg}}");
                // set the style for our bar
                bar.set_style(ProgressStyle::with_template(&style).unwrap());
                bar.enable_steady_tick(Duration::from_millis(Self::SPINNER_INTERVAL_MILLIS));
            }
            BarKind::Unbound => {
                // build our style string
                let style = format!("[{{elapsed_precise}}] {name} {{pos}} {{msg}}");
                // set the style for our bar
                bar.set_style(ProgressStyle::with_template(&style).unwrap());
            }
            BarKind::Bound(bound) => {
                // build our style string
                let style = format!(
                    "[{{elapsed_precise}}] {name} {{msg}} {{bar:40.cyan/blue}} {{pos:>7}}/{{len:7}} {{eta}} remaining"
                );
                // set the style for our bar
                bar.set_style(ProgressStyle::with_template(&style).unwrap());
                // set this bars length
                bar.set_length(bound);
                // start this bars progress at 0
                bar.set_position(0);
            }
            BarKind::UnboundIO | BarKind::IO => {
                // build our style string
                let style = format!(
                    "[{{elapsed_precise}}] {name} {{msg}} {{bytes}} {{binary_bytes_per_sec}}"
                );
                // set the style for our bar
                bar.set_style(ProgressStyle::with_template(&style).unwrap());
            }
        };
    }
}

/// The controller for multiple progress bars in Thorctl
#[derive(Default, Clone)]
pub struct MultiBar {
    /// The multiprogress controlling all our progress bars
    multi: MultiProgress,
}

impl MultiBar {
    /// Add child progress bar
    ///
    /// # Arguments
    ///
    /// * `name` - The name identifying this bar
    /// * `kind` - The kind of bar to add
    pub fn add(&self, name: &str, kind: BarKind) -> Bar {
        // create a new progress bar
        let bar = ProgressBar::new_spinner();
        // configure our new bar
        kind.setup(name, &bar);
        // add this bar to our multi progress bar
        self.multi.add(bar.clone());
        // build our child progress bar
        Bar {
            name: name.to_owned(),
            bar,
        }
    }

    /// Print an Error message
    ///
    /// # Arguments
    ///
    /// * `msg` - The error message to print
    pub fn error(&self, msg: &str) -> Result<(), Errors> {
        self.multi.println(msg)?;
        Ok(())
    }
}

/// A single progress bar in Thorctl
#[derive(Clone)]
pub struct Bar {
    /// The name of this progress bar
    name: String,
    /// The progress bar to use when showing progress
    pub bar: ProgressBar,
}

impl Bar {
    /// Create a new progress bar
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the progress bar
    /// * `kind` - The kind of progress bar to make
    pub fn new<T, M>(name: T, msg: M, kind: BarKind) -> Self
    where
        T: Into<String>,
        M: Into<Cow<'static, str>>,
    {
        let bar = Self {
            name: name.into(),
            bar: ProgressBar::new(0),
        };
        bar.refresh(msg, kind);
        bar
    }

    /// Create a simple progress timer with no bound or length
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the progress bar
    /// * `msg` - The message to display
    pub fn new_unbounded<T, M>(name: T, msg: M) -> Self
    where
        T: Into<String>,
        M: Into<Cow<'static, str>>,
    {
        let bar = Self {
            name: name.into(),
            bar: ProgressBar::new_spinner(),
        };
        bar.refresh(msg, BarKind::Unbound);
        bar
    }

    /// Rename this bar
    ///
    /// # Arguments
    ///
    /// * `name` - The name to use for this bar going forward
    pub fn rename(&mut self, name: String) {
        self.name = name;
    }

    /// Set a new message for this bar
    ///
    /// # Arguments
    ///
    /// * `msg` - The message to set
    pub fn set_message<M: Into<Cow<'static, str>>>(&self, msg: M) {
        // set our new message
        self.bar.set_message(msg);
    }

    /// Set the length for this bar
    ///
    /// # Arguments
    ///
    /// * `len` - The length to set
    #[allow(dead_code)]
    pub fn set_length(&self, len: u64) {
        self.bar.set_length(len);
    }

    /// Increment our total length
    ///
    /// # Arguments
    ///
    /// * `delta` - The delta to apply
    pub fn inc_length(&self, delta: u64) {
        self.bar.inc_length(delta);
    }

    /// Increment our progress
    ///
    /// # Arguments
    ///
    /// * `delta` - The delta to apply
    pub fn inc(&self, delta: u64) {
        self.bar.inc(delta);
    }

    /// Set the position of our progress bar
    ///
    /// # Arguments
    ///
    /// * `position` - The new position to set
    pub fn set_position(&self, position: u64) {
        self.bar.set_position(position);
    }

    /// Print an info message
    ///
    /// # Arguments
    ///
    /// * `msg` - The info message to print
    pub fn info<T: AsRef<str>>(&self, msg: T) {
        self.bar.println(format!(
            "{}: {} - {}",
            "Info".bright_blue(),
            &self.name,
            msg.as_ref(),
        ));
    }

    /// Print an info message without the bar's name included
    ///
    /// # Arguments
    ///
    /// * `msg` - The info message to print
    pub fn info_anonymous<T: AsRef<str>>(&self, msg: T) {
        self.bar
            .println(format!("{}: {}", "Info".bright_blue(), msg.as_ref(),));
    }

    /// Print a warning message
    ///
    /// # Arguments
    ///
    /// * `msg` - The warning message to print
    pub fn warning<T: AsRef<str>>(&self, msg: T) {
        self.bar.println(format!(
            "{}: {} - {}",
            "Warning".bright_yellow(),
            &self.name,
            msg.as_ref(),
        ));
    }

    /// Print an error message
    ///
    /// # Arguments
    ///
    /// * `msg` - The error message to print
    pub fn error<T: AsRef<str>>(&self, msg: T) {
        self.bar.println(format!(
            "{}: {} - {}",
            "Error".bright_red(),
            &self.name,
            msg.as_ref(),
        ));
    }

    /// Print an error message without the bar's name included
    ///
    /// # Arguments
    ///
    /// * `msg` - The error message to print
    pub fn error_anonymous<T: AsRef<str>>(&self, msg: T) {
        self.bar
            .println(format!("{}: {}", "Error".bright_red(), msg.as_ref(),));
    }

    /// Set a new message for this bar.
    ///
    /// This does not change the bars name.
    ///
    /// # Arguments
    ///
    /// * `msg` - The message to set
    pub fn refresh<M: Into<Cow<'static, str>>>(&self, msg: M, kind: BarKind) {
        // reset any progress in this bar
        self.bar.reset();
        // re-setup our bar
        kind.setup(&self.name, &self.bar);
        // set our new message
        self.bar.set_message(msg);
    }

    /// Finish this bar with an updated message
    pub fn finish_with_message<M: Into<Cow<'static, str>>>(&self, msg: M) {
        self.bar.finish_with_message(msg);
    }

    /// Finish this bar and clear it
    pub fn finish_and_clear(&self) {
        self.bar.finish_and_clear();
    }
}
