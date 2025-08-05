use std::fmt;

/// adds a text field to a multipart form
#[macro_export]
macro_rules! multipart_text {
    ($form:expr, $key:expr, $value:expr) => {
        match $value.take() {
            Some(value) => $form.text($key, value),
            None => $form,
        }
    };
}

/// adds a list of fields to a multipart form
#[macro_export]
macro_rules! multipart_list {
    ($form:expr, $key:expr, $value:expr) => {
        $value
            .drain(..)
            .fold($form, |form, value| form.text($key.to_owned(), value))
    };
}

/// reads in a file from a path and adds it to a form
#[macro_export]
macro_rules! multipart_file {
    ($form:expr, $key:expr, $path:expr) => {{
        // a path was set so read it an and set it to be upload
        let file = tokio::fs::File::open(&$path).await?;
        // get the length of this file so we can size our buffer correctly
        let len = file.metadata().await?.len();
        // convert our file into a framed read stream
        let codec = tokio_util::codec::BytesCodec::new();
        let stream = tokio_util::codec::FramedRead::new(file, codec);
        // convert our stream to a body to pass to reqwest
        let body = reqwest::Body::wrap_stream(stream);
        // build the form part that contains this file
        let file_part = reqwest::multipart::Part::stream_with_length(body, len)
            .mime_str("multipart/form-data")?;
        // try to set this files name
        let file_part = match $path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
        {
            Some(name) => file_part.file_name(name),
            None => file_part,
        };
        // add the file to upload
        let form = $form.part($key, file_part);
        form
    }};
}

/// Builds an error from creating a form for a struct
#[derive(Debug)]
pub struct FormError {
    /// The error message to return
    pub msg: String,
}

impl FormError {
    /// creates a new form error object
    ///
    /// # Arguments
    ///
    /// * `msg` - message to put in the error
    pub fn new(msg: String) -> FormError {
        FormError { msg }
    }
}

impl fmt::Display for FormError {
    /// Cast this error to either a string based on the message or the code
    ///
    /// # Arguments
    ///
    /// * `f` - The formatter that is being used
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl From<std::io::Error> for FormError {
    fn from(error: std::io::Error) -> Self {
        FormError::new(format!("IO Error {:#?}", error))
    }
}

impl From<reqwest::Error> for FormError {
    fn from(error: reqwest::Error) -> Self {
        FormError::new(format!("reqwest Error {:#?}", error))
    }
}

impl From<async_zip::error::ZipError> for FormError {
    fn from(error: async_zip::error::ZipError) -> Self {
        FormError::new(format!("Zip Error {:#?}", error))
    }
}

impl From<std::path::StripPrefixError> for FormError {
    fn from(error: std::path::StripPrefixError) -> Self {
        FormError::new(format!("Path Strip Prefix Error {:#?}", error))
    }
}
