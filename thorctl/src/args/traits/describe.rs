//! Traits and logic to allow for handlers to describe entities based on a user command

use std::io::{IsTerminal, Write};
use std::{io::BufRead, path::PathBuf};

use futures::StreamExt;
use owo_colors::OwoColorize;
use serde::{
    de::Deserialize,
    ser::{Serialize, SerializeSeq, Serializer},
};
use thorium::{Error, Thorium};
use tokio::io::AsyncBufReadExt;
use tokio_stream::wrappers::LinesStream;

use super::search::SearchParameterized;
use crate::{
    handlers::progress::{Bar, BarKind},
    utils,
};

/// The output format to use when describing entities
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum DescribeFormat {
    /// A styled, human-readable summary that renders the description markdown
    #[default]
    Human,
    /// The full entity details as JSON
    Json,
    // Yaml is intentionally reserved for a future addition
}

/// A sink that consumes each described datum (serialize it, print it, etc.)
type DatumSink<'a, D> = dyn FnMut(&D) -> Result<(), Error> + 'a;

/// A makeshift adapter trait providing a consistent interface to both
/// [`thorium::models::Cursor`] and [`thorium::client::Cursor`]
pub trait CursorLike<T> {
    /// Tell the cursor to get more data
    async fn more_data(&mut self) -> Result<(), Error>;

    /// Check if the cursor is exhausted
    fn is_exhausted(&self) -> bool;

    /// Drain the cursor's data buffer
    fn drain_data(&mut self) -> Vec<T>;
}

impl<T> CursorLike<T> for thorium::models::Cursor<T>
where
    for<'de> T: Deserialize<'de>,
    T: Serialize,
{
    /// Get the next page of data
    async fn more_data(&mut self) -> Result<(), Error> {
        self.refill().await
    }

    /// Return whether or not the cursor is exhausted
    fn is_exhausted(&self) -> bool {
        self.exhausted()
    }

    /// Drain the data stored within the cursor
    fn drain_data(&mut self) -> Vec<T> {
        self.data.drain(..).collect()
    }
}

impl<T> CursorLike<T> for thorium::client::Cursor<T>
where
    for<'de> T: Deserialize<'de>,
{
    /// Get the next page of data
    async fn more_data(&mut self) -> Result<(), Error> {
        self.next().await
    }

    /// Return whether or not the cursor is exhausted
    fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// Drain the details stored within the cursor; will return an empty Vec if
    /// details are not enabled
    fn drain_data(&mut self) -> Vec<T> {
        self.details.drain(..).collect()
    }
}

/// Attempt to unwrap datum or log an error and continue on failure
macro_rules! unwrap_datum_or_continue {
    ($datum:expr, $progress:expr) => {
        match $datum {
            Ok(datum) => datum,
            Err(err) => {
                let msg = err.msg().unwrap_or("An unknown error occurred".to_string());
                if let Some(progress) = $progress {
                    progress.error_anonymous(msg);
                } else {
                    eprintln!("{}: {}", "Error".bright_red(), msg);
                }
                continue;
            }
        }
    };
}

/// A private trait preventing inner [`DescribeCommand`] business logic
/// from being called outside of the module
pub trait DescribeSealed: SearchParameterized {
    /// The data to be described
    type Data: serde::Serialize + for<'b> serde::Deserialize<'b>;

    /// An individual target request for [`DescribeSealed::Data`]
    type Target<'a>;

    /// A search cursor (client or API-based)
    type Cursor: CursorLike<Self::Data>;

    /// Retrieve any raw targets the implementor might have
    fn raw_targets(&self) -> &[String];

    /// Print the resulting description JSON in a condensed format (no formatting/whitespace)
    fn condensed(&self) -> bool;

    /// Retrieve an optional description output file from the implementor
    fn out_path(&self) -> Option<&PathBuf>;

    /// Retrieve an optional path to a list of [`DescribeSealed::Target`] or data
    /// that can be converted to such
    fn target_list(&self) -> Option<&PathBuf>;

    /// Parse a target from a raw str
    fn parse_target<'a>(&self, raw: &'a str) -> Result<Self::Target<'a>, thorium::Error>;

    /// Query for an individual [`DescribeSealed::Target`]'s [`DescribeSealed::Data`]
    /// using the [`Thorium`] client
    ///
    /// # Arguments
    ///
    /// * `target` - The target to query data for
    /// * `thorium` - The Thorium client
    async fn retrieve_data(
        &self,
        target: Self::Target<'_>,
        thorium: &Thorium,
    ) -> Result<Self::Data, thorium::Error>;

    /// Query for a cursor/cursors of [`DescribeSealed::Data`] using a [`Thorium`] client
    /// based on the implementor's [`SearchParams`] as described in [`SearchParameterized`];
    /// possibly return multiple cursors if the [`CursorLike`] requires multiple queries
    /// for certain parameters.
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    async fn retrieve_data_search(
        &self,
        thorium: &Thorium,
    ) -> Result<Vec<Self::Cursor>, thorium::Error>;

    /// The output format to use when describing this command's entities
    ///
    /// Defaults to JSON; commands that support human-readable output override
    /// this (and [`DescribeSealed::render_human`]). Implementors return an error
    /// when the user requests a format the command can't produce.
    fn format(&self) -> Result<DescribeFormat, Error> {
        Ok(DescribeFormat::Json)
    }

    /// Render a single datum as human-readable text into `out`
    ///
    /// Styling is emitted only when `ansi` is set. Commands that return
    /// [`DescribeFormat::Human`] from [`DescribeSealed::format`] must override
    /// this; the default errors.
    ///
    /// # Arguments
    ///
    /// * `datum` - The datum to render
    /// * `out` - The writer to render into
    /// * `ansi` - Whether to style the output with ANSI escape codes
    fn render_human(
        &self,
        _datum: &Self::Data,
        _out: &mut dyn Write,
        _ansi: bool,
    ) -> Result<(), Error> {
        Err(Error::new(
            "human-readable output is not supported for this command",
        ))
    }

    /// Calculate a bound on how much data will be retrieved if possible
    fn calculate_bound(&self) -> Result<Option<usize>, Error> {
        let mut count = 0;
        if self.has_parameters() || self.apply_to_all() {
            let params = self.get_search_params();
            if params.no_limit {
                return Ok(None);
            }
            count += params.limit;
        }
        if let Some(list_path) = self.target_list() {
            let reader = std::io::BufReader::new(std::fs::File::open(list_path)?);
            count += reader
                .lines()
                .try_fold(0, |count, line| line.map(|_| count + 1))?;
        }
        count += self.raw_targets().len();
        Ok(Some(count))
    }

    /// Describe specific targets given to the implementor
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    /// * `on_datum` - The sink to hand each retrieved datum to
    /// * `progress` - An optional progress bar to show description progress
    async fn describe_targets(
        &self,
        thorium: &Thorium,
        on_datum: &mut DatumSink<'_, Self::Data>,
        progress: Option<&Bar>,
    ) -> Result<usize, Error> {
        let mut num_described = 0;
        // map each target to a future that retrieves the data from Thorium
        let futures = self.raw_targets().iter().map(|raw_target| async {
            let datum = self
                .retrieve_data(self.parse_target(raw_target)?, thorium)
                .await?;
            Ok::<Self::Data, Error>(datum)
        });
        // buffer the data to prevent storing too much at once
        let mut buffered_data = futures::stream::iter(futures).buffered(10000);
        // hand each datum we retrieve to the sink
        while let Some(datum) = buffered_data.next().await {
            let datum = unwrap_datum_or_continue!(datum, progress);
            on_datum(&datum)?;
            num_described += 1;
            if let Some(progress) = progress {
                progress.inc(1);
            }
        }
        Ok(num_described)
    }

    /// Describe targets given to the implementor in a list file
    ///
    /// # Arguments
    ///
    /// * `list_path` - The path to the file containing the list of targets to describe
    /// * `thorium` - The Thorium client
    /// * `on_datum` - The sink to hand each retrieved datum to
    /// * `progress` - An optional progress bar
    async fn describe_list(
        &self,
        list_path: &PathBuf,
        thorium: &Thorium,
        on_datum: &mut DatumSink<'_, Self::Data>,
        progress: Option<&Bar>,
    ) -> Result<usize, Error> {
        let mut num_described = 0;
        // open the list file and create a stream of the file's lines
        let list_file = tokio::fs::File::open(list_path).await.map_err(|err| {
            Error::new(format!(
                "Error opening list file '{}': {err}",
                list_path.to_string_lossy(),
            ))
        })?;
        let lines = tokio::io::BufReader::new(list_file).lines();
        // map each line to a Thorium API request for data
        let mut data_stream = LinesStream::new(lines)
            .map(|line| async move {
                let line = line?;
                let target = self
                    .parse_target(line.as_str())
                    .map_err(|_| Error::new(format!("Malformed line '{line}'!")))?;
                self.retrieve_data(target, thorium).await
            })
            .buffered(10000);
        while let Some(datum) = data_stream.next().await {
            let datum = unwrap_datum_or_continue!(datum, progress);
            on_datum(&datum)?;
            num_described += 1;
            if let Some(progress) = progress {
                progress.inc(1);
            }
        }
        Ok(num_described)
    }

    /// Describe targets found in a search
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    /// * `on_datum` - The sink to hand each retrieved datum to
    /// * `progress` - An optional progress bar
    async fn describe_search(
        &self,
        thorium: &Thorium,
        on_datum: &mut DatumSink<'_, Self::Data>,
        progress: Option<&Bar>,
    ) -> Result<usize, Error> {
        // track the number described
        let mut num_described = 0;
        // get the data cursor
        let cursors = self.retrieve_data_search(thorium).await?;
        for mut cursor in cursors {
            loop {
                let data: Vec<Self::Data> = cursor.drain_data();
                for datum in &data {
                    // hand the retrieved data to the sink
                    on_datum(datum)?;
                }
                num_described += data.len();
                if let Some(progress) = progress {
                    progress.inc(data.len() as u64);
                }
                if cursor.is_exhausted() {
                    break;
                }
                cursor.more_data().await?;
            }
        }
        Ok(num_described)
    }

    /// A general function to describe all data given to the implementor
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    /// * `on_datum` - The sink to hand each retrieved datum to
    /// * `progress` - An optional progress bar
    async fn describe_sealed(
        &self,
        thorium: &Thorium,
        on_datum: &mut DatumSink<'_, Self::Data>,
        progress: Option<&Bar>,
    ) -> Result<usize, Error> {
        let mut num_described = 0;
        if self.has_targets() {
            // describe any specific targets
            num_described += self.describe_targets(thorium, on_datum, progress).await?;
        }
        if let Some(target_list_path) = &self.target_list() {
            // describe targets given in a list file
            num_described += self
                .describe_list(target_list_path, thorium, on_datum, progress)
                .await?;
        }
        if self.has_parameters() || self.apply_to_all() {
            // describe targets using a search
            num_described += self.describe_search(thorium, on_datum, progress).await?;
        }
        Ok(num_described)
    }
}

/// A command that can describe some kind of data in Thorium
pub trait DescribeCommand: DescribeSealed {
    /// Describe data, retrieving it from the Thorium API, serializing it, then
    /// outputting to either stdout or an output file
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    async fn describe(&self, thorium: &Thorium) -> Result<(), Error> {
        // track progress if we're writing to an output file
        let progress = if let Some(out_path) = self.out_path() {
            // prepend the current directory ("./" or ".\") to emphasize this is the current path
            let displayed_output = utils::fs::prepend_current_dir(out_path);
            match self.calculate_bound() {
                Ok(bound_opt) => match bound_opt {
                    Some(bound) => Some(Bar::new(
                        format!("Writing description to '{displayed_output}':"),
                        "",
                        BarKind::Bound(bound as u64),
                    )),
                    None => Some(Bar::new(
                        format!("Writing description to '{displayed_output}':"),
                        "",
                        BarKind::Unbound,
                    )),
                },
                Err(err) => {
                    return Err(Error::new(format!(
                        "Error calculating bounds on data collection: {err}"
                    )));
                }
            }
        } else {
            None
        };
        self.validate_search()?;
        let out_file = match self.out_path() {
            Some(out_path) => {
                if let Some(parent) = out_path.parent() {
                    tokio::fs::create_dir_all(parent).await.map_err(|err| {
                        Error::new(format!(
                            "Error creating parent directory for output file '{}': {err}",
                            out_path.to_string_lossy(),
                        ))
                    })?;
                }
                match std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(out_path)
                {
                    Ok(file) => Some(file),
                    Err(err) => {
                        return Err(Error::new(format!(
                            "Unable to open file '{}': {err}",
                            out_path.to_string_lossy()
                        )));
                    }
                }
            }
            None => None,
        };
        // `--condensed` is a JSON modifier, so it forces JSON regardless of format
        let format = if self.condensed() {
            DescribeFormat::Json
        } else {
            self.format()?
        };
        // unify the stdout and file destinations into a single writer
        let to_stdout = out_file.is_none();
        let mut writer: Box<dyn Write> = match out_file {
            Some(file) => Box::new(file),
            None => Box::new(std::io::stdout()),
        };
        // describe the data, saving the actual number described in case it doesn't match the bound
        let num_described = match format {
            DescribeFormat::Json => {
                // serialize each datum into the JSON sequence as it arrives
                let num_described = if self.condensed() {
                    let mut ser = serde_json::Serializer::new(&mut *writer);
                    let mut seq = ser.serialize_seq(None)?;
                    // scope the sink so its borrow on `seq` ends before `seq.end`
                    let num = {
                        let mut sink = |datum: &Self::Data| {
                            seq.serialize_element(datum)
                                .map_err(|err| Error::new(format!("Error serializing data: {err}")))
                        };
                        self.describe_sealed(thorium, &mut sink, progress.as_ref())
                            .await?
                    };
                    seq.end()?;
                    num
                } else {
                    let mut ser = serde_json::Serializer::pretty(&mut *writer);
                    let mut seq = ser.serialize_seq(None)?;
                    // scope the sink so its borrow on `seq` ends before `seq.end`
                    let num = {
                        let mut sink = |datum: &Self::Data| {
                            seq.serialize_element(datum)
                                .map_err(|err| Error::new(format!("Error serializing data: {err}")))
                        };
                        self.describe_sealed(thorium, &mut sink, progress.as_ref())
                            .await?
                    };
                    seq.end()?;
                    num
                };
                // serde doesn't add a trailing newline, so add one for the terminal
                if to_stdout {
                    writeln!(writer)
                        .map_err(|err| Error::new(format!("Error writing output: {err}")))?;
                }
                num_described
            }
            DescribeFormat::Human => {
                // only emit ANSI styling for an interactive terminal
                let ansi = to_stdout && std::io::stdout().is_terminal();
                let mut first = true;
                let mut sink = |datum: &Self::Data| -> Result<(), Error> {
                    if first {
                        first = false;
                    } else {
                        // separate consecutive entities with a rule
                        writeln!(writer, "\n{}\n", utils::render::separator(ansi))
                            .map_err(|err| Error::new(format!("Error writing output: {err}")))?;
                    }
                    self.render_human(datum, &mut *writer, ansi)?;
                    writeln!(writer)
                        .map_err(|err| Error::new(format!("Error writing output: {err}")))?;
                    Ok(())
                };
                self.describe_sealed(thorium, &mut sink, progress.as_ref())
                    .await?
            }
        };
        // finish the progress bar if we have one
        if let Some(progress) = progress {
            progress.set_length(num_described as u64);
            progress.finish_with_message("Done!");
        }
        Ok(())
    }
}
