//! Parses an email and dumps out results
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use clap::Parser;
use mail_parser::{
    Address, ContentType, HeaderValue, Message, MessageParser, MessagePart, MimeHeaders, PartType,
    parsers::message::IntoByteSlice,
};

mod args;

/// A tag add helper
fn add_tag<K: Into<String>, V: Into<String>>(
    tags: &mut HashMap<String, Vec<String>>,
    key: K,
    value: V,
) {
    // get an entry to our basic email tags
    let entry = tags.entry(key.into()).or_default();
    // add the has attachments tag
    entry.push(value.into());
}

/// A tag add helper
fn add_opt_tag<K: Into<String>, V: Into<String>>(
    tags: &mut HashMap<String, Vec<String>>,
    key: K,
    opt_value: Option<V>,
) {
    // if we have a value then set this tag
    if let Some(value) = opt_value {
        // get an entry to our basic email tags
        let entry = tags.entry(key.into()).or_default();
        // add the has attachments tag
        entry.push(value.into());
    }
}

/// An email address
#[derive(Debug, Serialize, Deserialize)]
pub struct EmailAddress<'a> {
    /// The name for this email
    pub name: Option<Cow<'a, str>>,
    /// The email address for this user
    pub email: Option<Cow<'a, str>>,
}

impl<'a> EmailAddress<'a> {
    /// Build a list of email address from parsed addresses
    pub fn from_addr(
        raw_addrs: Option<&Address<'a>>,
        tags: &mut HashMap<String, Vec<String>>,
        tag_key: &str,
    ) -> Vec<Self> {
        // iterate over the addrs in this address set
        match raw_addrs {
            Some(raw_addrs) => {
                // start with an empty list of addrs
                let mut addresses = Vec::with_capacity(1);
                // iterate over all the emails
                for raw_addr in raw_addrs.iter() {
                    // add either the address or the name to this entry
                    let tag_value = match (&raw_addr.address, &raw_addr.name) {
                        (Some(addr), _) => addr.to_string(),
                        (_, Some(name)) => name.to_string(),
                        _ => {
                            println!("No name or addr: {raw_addr:#?}?");
                            // continue on to the next addr
                            continue;
                        }
                    };
                    // add our tag value
                    add_tag(tags, "EmailAddress", &tag_value);
                    add_tag(tags, tag_key, tag_value);
                    // build an email addr object
                    let addr = EmailAddress {
                        name: raw_addr.name.clone(),
                        email: raw_addr.address.clone(),
                    };
                    addresses.push(addr);
                }
                addresses
            }
            None => vec![],
        }
    }
}

/// Extract tags from this parsed email's headers
fn extract_header_tag<'a>(name: &str, value: &HeaderValue<'a>, entry: &mut Vec<String>) {
    // parse our header value correctly
    match value {
        HeaderValue::Text(value) => entry.push(format!("{name}={value}")),
        HeaderValue::TextList(values) => {
            // convert our text list to
            entry.extend(values.iter().map(|val| format!("{name}={val}")))
        }
        HeaderValue::DateTime(datetime) => {
            // convert our datetime to a string
            entry.push(format!("{}={}", name, datetime.to_rfc3339()))
        }
        HeaderValue::Address(addrs) => {
            // convert our address to strings
            let addr_iter = addrs
                .iter()
                .filter_map(|addr| addr.address())
                .map(|addr| format!("{name}={addr}"));
            // extend our header value list with these addresses
            entry.extend(addr_iter);
        }
        HeaderValue::ContentType(ctype) => {
            // build a condensed string for this content type
            let condensed = match &ctype.c_subtype {
                Some(subtype) => format!("{}={}/{}", name, ctype.c_type, subtype),
                None => format!("{}={}", name, ctype.c_type),
            };
            // add our condensed content type
            entry.push(condensed);
        }
        HeaderValue::Received(rx) => match rx.from() {
            Some(from) => entry.push(format!("{}={}", name, from)),
            None => panic!("No host/ip in Recieved header?"),
        },
        HeaderValue::Empty => println!("Found empty header: {}", name),
    }
}

/// An attachment to this email
#[derive(Debug, Serialize, Deserialize)]
pub struct EmailAttachment {
    /// The name for this attachment if one was set
    pub name: Option<String>,
    /// The content type for this attachment
    pub content_type: Option<String>,
    /// Whether this attachment is text or not
    pub is_text: bool,
}

impl EmailAttachment {
    /// Parse an invidividual attachment from an email
    async fn parse_attachment<'a>(
        index: usize,
        attachment: &MessagePart<'a>,
        tags: &mut HashMap<String, Vec<String>>,
        path: &PathBuf,
    ) -> Self {
        // get this attachments name
        let name = attachment.attachment_name().map(|name| name.to_owned());
        // get our content type if we have one
        let ctype = match attachment.content_type() {
            Some(ctype) => {
                // build a condensed string for this content type
                match &ctype.subtype() {
                    Some(subtype) => Some(format!("{}/{}", ctype.c_type, subtype)),
                    None => Some(format!("{}", ctype.c_type)),
                }
            }
            None => None,
        };
        // add our attachment tags
        add_opt_tag(tags, "EmailAttachmentName", name.as_ref());
        add_opt_tag(tags, "EmailAttachmentContentType", ctype.as_ref());
        // get this attachments data
        let child_data = match &attachment.body {
            PartType::Text(text) => Some(text.as_bytes()),
            PartType::Binary(binary) => Some(binary.into_byte_slice()),
            PartType::Html(html) => Some(html.as_bytes()),
            PartType::InlineBinary(inline) => Some(inline.into_byte_slice()),
            // not sure how we should handle multipart attachments
            // one attachment maps to multiple attachments and I am
            // not sure where/how to retrieve them right now
            PartType::Multipart(multipart) => {
                println!("{index} has multipart with {multipart:?}!!");
                add_tag(tags, "EmailHas", "MultipartAttachments");
                None
            }
            PartType::Message(message) => Some(message.raw_message()),
        };
        // if we have child datat then write it to disk
        if let Some(child_bytes) = child_data {
            // get our attachment name or use a default for our file name
            let child_path = match &name {
                Some(name) => path.join(name),
                None => path.join(format!("attachment_{index}")),
            };
            // write this attachment to disk
            tokio::fs::write(child_path, child_bytes).await.unwrap();
        }
        // build our email attachment object
        EmailAttachment {
            name,
            content_type: ctype,
            is_text: attachment.is_text(),
        }
    }
    /// Convert an email to a list of parsed attachments
    pub async fn from_email<'a>(
        email: &Message<'a>,
        tags: &mut HashMap<String, Vec<String>>,
        attachment_path: &PathBuf,
    ) -> Vec<Self> {
        // The list of parsed attachments
        let mut parsed = Vec::with_capacity(email.attachments.len());
        // step over and parse our email attachments
        for (index, attachment) in email.attachments().enumerate() {
            // parse this attachment
            let parsed_attachment =
                Self::parse_attachment(index, attachment, tags, attachment_path).await;
            // add this parsed attachment
            parsed.push(parsed_attachment);
        }
        // if we found attachments then add that tag
        if !parsed.is_empty() {
            // add our has attachment tag
            add_tag(tags, "Email", "HasAttachments");
        }
        parsed
    }
}

#[derive(Debug, Serialize)]
pub struct EmailBodies {
    /// The text bodies for this email
    text: Vec<String>,
    /// The html bodies for this email
    html: Vec<String>,
}

impl EmailBodies {
    /// Extract the bodies from an email that we extract
    ///
    /// # Arguments
    ///
    /// * `email` - The email to extract bodies from
    /// * `tags` - The tags for this email analsysis
    fn new(email: &Message, tags: &mut HashMap<String, Vec<String>>) -> Self {
        // get our text bodies
        let text: Vec<String> = email
            .text_bodies()
            .filter_map(|body| body.text_contents())
            .map(ToOwned::to_owned)
            .collect();
        println!("Found {} text bodies", text.len());
        // get our html bodies
        let html: Vec<String> = email
            .html_bodies()
            .filter_map(|body| body.text_contents())
            .map(ToOwned::to_owned)
            .collect();
        println!("Found {} html bodies", text.len());
        // check if we didn't find the same number of html/text bodies
        if text.len() != html.len() {
            // log that we didn't find the same number of html/text bodies
            println!("text/html body count mismatch!");
            // add a tag for the mismatch
            add_tag(tags, "Email", "TextHtmlBodyCountMismatch");
        }
        // return our email bodies
        EmailBodies { text, html }
    }
}

#[derive(Debug, Serialize)]
pub struct EmailResults<'a> {
    /// The addresses that sent this email (generally one but could be many)
    pub from: Vec<EmailAddress<'a>>,
    /// The address an email was sent too
    pub to: Vec<EmailAddress<'a>>,
    /// Who this email was cc'd too
    pub cc: Vec<EmailAddress<'a>>,
    /// Who this email was bcc'd too
    pub bcc: Vec<EmailAddress<'a>>,
    /// This emails subject
    pub subject: Option<String>,
    /// The headers for this email
    pub headers: HashMap<String, Vec<HeaderValue<'a>>>,
    /// The attachments for this email
    pub attachments: Vec<EmailAttachment>,
    /// The bodies for this email
    pub bodies: EmailBodies,
    /// The tags to supply to thorium for this email
    #[serde(skip_serializing)]
    pub tags: HashMap<String, Vec<String>>,
}

impl<'a> EmailResults<'a> {
    /// Parse an email
    pub async fn new(raw_email: &'a str, attachment_path: &PathBuf) -> Self {
        // parse this email
        println!("Parsing email");
        let email = MessageParser::new().parse(raw_email).unwrap();
        // build a map of tags to supply to Thorium as we go
        let mut tags = HashMap::with_capacity(20);
        // get our from address
        // get an entry to our from tags
        let from = EmailAddress::from_addr(email.from(), &mut tags, "EmailFrom");
        // get our to address
        // get an entry to our to tags
        let to = EmailAddress::from_addr(email.to(), &mut tags, "EmailTo");
        println!("Found {} from addresses", from.len());
        println!("Found {} to addresses", to.len());
        // get the cc/bcc values
        let cc = EmailAddress::from_addr(email.cc(), &mut tags, "EmailCC");
        let bcc = EmailAddress::from_addr(email.bcc(), &mut tags, "EmailBCC");
        println!("Found {} cc addresses", cc.len());
        println!("Found {} bcc addresses", bcc.len());
        // get this emails subject
        let subject = email.subject().map(ToOwned::to_owned);
        // add our subject as a tag
        add_opt_tag(&mut tags, "EmailSubject", subject.as_ref());
        // get this emails attachments
        let attachments = EmailAttachment::from_email(&email, &mut tags, attachment_path).await;
        println!("Found {} attachments", attachments.len());
        // build a map of header info
        let mut headers = HashMap::with_capacity(email.headers().len());
        // tack how many headers we find
        let mut header_cnt = 0;
        // step over the headers for this email
        for header in email.headers() {
            // get our header name as a string
            let name = header.name();
            //// extract any tags from this header
            //extract_header_tag(&name, &header.value(), entry);
            // add this header value to our results
            let header_entry: &mut Vec<HeaderValue<'a>> =
                headers.entry(name.to_owned()).or_default();
            // get this headers value
            let value = header.value().clone();
            // add this header value to our header map
            header_entry.push(value);
            // increment our found header count
            header_cnt += 1;
        }
        println!("Found {header_cnt} headers");
        // get our email bodies
        let bodies = EmailBodies::new(&email, &mut tags);
        EmailResults {
            from,
            to,
            cc,
            bcc,
            subject,
            headers,
            attachments,
            bodies,
            tags,
        }
    }
}

#[tokio::main]
async fn main() {
    // get our args
    let args = args::Args::parse();
    // read this email from disk
    let raw_email = tokio::fs::read_to_string(&args.path).await.unwrap();
    // parse the email
    let results = EmailResults::new(&raw_email, &args.attachments_output).await;
    // write our results out to disk as serialized json
    println!("Writing results to {}", args.output.display());
    let serialized_results = serde_json::to_vec(&results).unwrap();
    tokio::fs::write(&args.output, &serialized_results)
        .await
        .unwrap();
    // serialize and write our tags to disk
    println!("Writing tags to {}", args.tags_output.display());
    let serialized_tags = serde_json::to_vec(&results.tags).unwrap();
    tokio::fs::write(&args.tags_output, &serialized_tags)
        .await
        .unwrap();
}
