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
    IO(u64),
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
            BarKind::UnboundIO => {
                // build our style string
                let style = format!(
                    "[{{elapsed_precise}}] {name} {{msg}} {{bytes}} {{binary_bytes_per_sec}}"
                );
                // set the style for our bar
                bar.set_style(ProgressStyle::with_template(&style).unwrap());
                // start this bars progress at 0
                bar.set_position(0);
            }
            BarKind::IO(bound) => {
                // build our style string
                let style = format!(
                    "[{{elapsed_precise}}] {name} {{msg}} {{bytes}}/{{total_bytes}} {{binary_bytes_per_sec}}"
                );
                // set this bars length
                bar.set_length(bound);
                // start this bars progress at 0
                bar.set_position(0);
                // set the style for our bar
                bar.set_style(ProgressStyle::with_template(&style).unwrap());
            }
        };
    }
}

/// The controller for multiple progress bars in Thorctl
#[derive(Clone)]
pub struct MultiBar {
    /// The multiprogress controlling all our progress bars
    multi: Option<MultiProgress>,
}

impl MultiBar {
    /// Create a new multi progress bar
    ///
    /// # Arguments
    ///
    /// * `quiet` - Whether this progress bar should be visible
    pub fn new(quiet: bool) -> Self {
        // this progress bar is not quiet then create a progress bar
        let multi = if quiet {
            None
        } else {
            Some(MultiProgress::new())
        };
        MultiBar { multi }
    }

    /// Add child progress bar
    ///
    /// # Arguments
    ///
    /// * `name` - The name identifying this bar
    /// * `kind` - The kind of bar to add
    pub fn add(&self, name: &str, kind: BarKind) -> Bar {
        // if quiet is set then don't create a bar
        let bar = match &self.multi {
            Some(multi) => {
                // create a new progress bar
                let bar = ProgressBar::new_spinner();
                // configure our new bar
                kind.setup(name, &bar);
                // add this bar to our multi progress bar
                multi.add(bar.clone());
                // return our child bar
                Some(bar)
            }
            None => None,
        };
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
        match &self.multi {
            Some(multi) => multi.println(msg)?,
            None => eprintln!("{msg}"),
        }
        Ok(())
    }
}

/// A single progress bar in Thorctl
#[derive(Clone)]
pub struct Bar {
    /// The name of this progress bar
    name: String,
    /// The progress bar to use when showing progress
    pub bar: Option<ProgressBar>,
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
            bar: Some(ProgressBar::new(0)),
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
            bar: Some(ProgressBar::new_spinner()),
        };
        bar.refresh(msg, BarKind::Unbound);
        bar
    }

    /// Change this bars style
    ///
    /// # Arguments
    ///
    /// * `style` - The style to set
    pub fn set_style(&self, style: ProgressStyle) {
        // check if we are in quiet mode or not
        if let Some(bar) = &self.bar {
            // update our bars style
            bar.set_style(style);
        }
    }

    /// Set a steady tick rate for our bar
    ///
    /// # Arguments
    ///
    /// * `tick` - The tick duration to set
    pub fn enable_steady_tick(&self, tick: Duration) {
        // check if we are in quiet mode or not
        if let Some(bar) = &self.bar {
            bar.enable_steady_tick(tick);
        }
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
        // check if we are in quiet mode or not
        if let Some(bar) = &self.bar {
            // set our new message
            bar.set_message(msg);
        }
    }

    /// Set the length for this bar
    ///
    /// # Arguments
    ///
    /// * `len` - The length to set
    #[allow(dead_code)]
    pub fn set_length(&self, len: u64) {
        // check if we are in quiet mode or not
        if let Some(bar) = &self.bar {
            bar.set_length(len);
        }
    }

    /// Increment our total length
    ///
    /// # Arguments
    ///
    /// * `delta` - The delta to apply
    pub fn inc_length(&self, delta: u64) {
        // check if we are in quiet mode or not
        if let Some(bar) = &self.bar {
            bar.inc_length(delta);
        }
    }

    /// Increment our progress
    ///
    /// # Arguments
    ///
    /// * `delta` - The delta to apply
    pub fn inc(&self, delta: u64) {
        // check if we are in quiet mode or not
        if let Some(bar) = &self.bar {
            bar.inc(delta);
        }
    }

    /// Set the position of our progress bar
    ///
    /// # Arguments
    ///
    /// * `position` - The new position to set
    pub fn set_position(&self, position: u64) {
        // check if we are in quiet mode or not
        if let Some(bar) = &self.bar {
            bar.set_position(position);
        }
    }

    /// Print an info message
    ///
    /// # Arguments
    ///
    /// * `msg` - The info message to print
    pub fn info<T: AsRef<str>>(&self, msg: T) {
        // check if we are in quiet mode or not
        if let Some(bar) = &self.bar {
            bar.println(format!(
                "{}: {} - {}",
                "Info".bright_blue(),
                &self.name,
                msg.as_ref(),
            ));
        }
    }

    /// Print an info message without the bar's name included
    ///
    /// # Arguments
    ///
    /// * `msg` - The info message to print
    pub fn info_anonymous<T: AsRef<str>>(&self, msg: T) {
        // check if we are in quiet mode or not
        if let Some(bar) = &self.bar {
            bar.println(format!("{}: {}", "Info".bright_blue(), msg.as_ref(),));
        }
    }

    /// Print a warning message
    ///
    /// # Arguments
    ///
    /// * `msg` - The warning message to print
    pub fn warning<T: AsRef<str>>(&self, msg: T) {
        // check if we are in quiet mode or not
        if let Some(bar) = &self.bar {
            bar.println(format!(
                "{}: {} - {}",
                "Warning".bright_yellow(),
                &self.name,
                msg.as_ref(),
            ));
        }
    }

    /// Print an error message
    ///
    /// # Arguments
    ///
    /// * `msg` - The error message to print
    pub fn error<T: AsRef<str>>(&self, msg: T) {
        // print our error if we are in quiet mode or not
        match &self.bar {
            Some(bar) => {
                bar.println(format!(
                    "{}: {} - {}",
                    "Error".bright_red(),
                    &self.name,
                    msg.as_ref(),
                ));
            }
            None => {
                eprintln!(
                    "{}: {} - {}",
                    "Error".bright_red(),
                    &self.name,
                    msg.as_ref(),
                )
            }
        }
    }

    /// Print an error message without the bar's name included
    ///
    /// # Arguments
    ///
    /// * `msg` - The error message to print
    pub fn error_anonymous<T: AsRef<str>>(&self, msg: T) {
        // print our error if we are in quiet mode or not
        match &self.bar {
            Some(bar) => {
                bar.println(format!("{}: {}", "Error".bright_red(), msg.as_ref(),));
            }
            None => {
                eprintln!("{}: {}", "Error".bright_red(), msg.as_ref());
            }
        }
    }

    /// Set a new message for this bar.
    ///
    /// This does not change the bars name.
    ///
    /// # Arguments
    ///
    /// * `msg` - The message to set
    pub fn refresh<M: Into<Cow<'static, str>>>(&self, msg: M, kind: BarKind) {
        // check if we are in quiet mode or not
        if let Some(bar) = &self.bar {
            // reset any progress in this bar
            bar.reset();
            // resetup our bar
            kind.setup(&self.name, &bar);
            // set our new message
            bar.set_message(msg);
        }
    }

    /// Finish the bar, leaving its message displayed
    pub fn finish(&self) {
        // check if we are in quiet mode or not
        if let Some(bar) = &self.bar {
            bar.finish();
        }
    }

    /// Finish this bar with an updated message
    pub fn finish_with_message<M: Into<Cow<'static, str>>>(&self, msg: M) {
        // check if we are in quiet mode or not
        if let Some(bar) = &self.bar {
            bar.finish_with_message(msg);
        }
    }

    /// Finish this bar and clear it
    pub fn finish_and_clear(&self) {
        // check if we are in quiet mode or not
        if let Some(bar) = &self.bar {
            bar.finish_and_clear();
        }
    }

    /// Suspend the progress bar while the function `f` is executing
    pub fn suspend<F: FnOnce() -> R, R>(&self, f: F) -> R {
        // check if we are in quiet mode or not
        match &self.bar {
            Some(bar) => bar.suspend(f),
            None => f(),
        }
    }
}
