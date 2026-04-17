//! Different kinds of network activity

use chrono::prelude::*;
use std::net::IpAddr;
use std::str::FromStr;

use crate::models::InvalidEnum;

#[cfg(feature = "client")]
use crate::{multipart_text, multipart_text_to_string};

/// The different states for a network connection
#[derive(Debug, Clone, Copy, Serialize, Deserialize, strum::Display)]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreAsStr))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum NetConState {
    /// This conneciton is waiting for a external connection request
    Listen,
    /// This connection has sent a connection request and is waiting for a response
    Syn,
    /// This connection has sent a Syn packet and received a sync-ack
    SynAck,
    /// This connection is open and can send and receive data
    Established,
    /// One side has sent a closure request and is waiting for an acknowledgement
    Fin,
    /// The destination side has requested to terminate this connection
    CloseWait,
    /// The destination side is waiting for the final acknowledgement after sending its own fin packet
    LastAck,
    /// This connection is waiting for the destination to receive a connection termination request
    TimeWait,
    /// This connection has closed
    Closed,
}

impl NetConState {
    /// Convert this [`NetConState`] into a str
    pub fn as_str(&self) -> &'static str {
        match self {
            NetConState::Listen => "Listen",
            NetConState::Syn => "Syn",
            NetConState::SynAck => "SynAck",
            NetConState::Established => "Established",
            NetConState::Fin => "Fin",
            NetConState::CloseWait => "CloseWait",
            NetConState::LastAck => "LastAck",
            NetConState::TimeWait => "TimeWait",
            NetConState::Closed => "Closed",
        }
    }
}

impl FromStr for NetConState {
    type Err = InvalidEnum;

    /// Cast a str to an [`NetConState`]
    ///
    /// # Arguments
    ///
    /// * `val` - The str to cast
    fn from_str(val: &str) -> Result<Self, Self::Err> {
        match val {
            "Listen" => Ok(NetConState::Listen),
            "Syn" => Ok(NetConState::Syn),
            "SynAck" => Ok(NetConState::SynAck),
            "Established" => Ok(NetConState::Established),
            "Fin" => Ok(NetConState::Fin),
            "CloseWait" => Ok(NetConState::CloseWait),
            "LastAck" => Ok(NetConState::LastAck),
            "TimeWait" => Ok(NetConState::TimeWait),
            "Closed" => Ok(NetConState::Closed),
            _ => Err(InvalidEnum(format!("Unknown enum variant: {val}"))),
        }
    }
}

/// The different protocols at the transport layer
///
/// This is not exhaustive and more protocols should be added as needed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, strum::Display)]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreAsStr))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum TransportLayerProtocol {
    /// A protocol for reliable ordered delivery of data
    TCP,
    /// A lightweight protocol without error correction or acknowledgement
    UDP,
    /// A reliable message oriented protocol supporting multi-streaming and multi-homing
    SCTP,
}

impl TransportLayerProtocol {
    /// Convert this [`TransportLayerProtocol`] into a str
    pub fn as_str(&self) -> &'static str {
        match self {
            TransportLayerProtocol::TCP => "TCP",
            TransportLayerProtocol::UDP => "UDP",
            TransportLayerProtocol::SCTP => "SCTP",
        }
    }
}

impl FromStr for TransportLayerProtocol {
    type Err = InvalidEnum;

    /// Cast a str to an [`TransportLayerProtocol`]
    ///
    /// # Arguments
    ///
    /// * `val` - The str to cast
    fn from_str(val: &str) -> Result<Self, Self::Err> {
        match val {
            "tcp" | "TCP" | "TCPv4" | "TCPv6" => Ok(TransportLayerProtocol::TCP),
            "udp" | "UDP" | "UDPv4" | "UDPv6" => Ok(TransportLayerProtocol::UDP),
            "sctp" | "SCTP" | "SCTPv4" | "SCTPv6" => Ok(TransportLayerProtocol::SCTP),
            _ => Err(InvalidEnum(format!("Unknown enum variant: {val}"))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NetworkConnection {
    /// The protocol this network connection is using at the transport layer
    pub protocol: Option<TransportLayerProtocol>,
    /// The source address for this connection
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub source: IpAddr,
    /// source port for this connection
    pub source_port: Option<u16>,
    /// The destination address for this connection
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub destination: IpAddr,
    /// The destination port for this connection
    pub destination_port: u16,
    /// The state of this connection
    pub state: Option<NetConState>,
    /// The pid this process is from
    pub pid: Option<u64>,
    /// The name of this process that owns this connection
    pub process: Option<String>,
    /// When this process was created (not in Thorium but the actually network connection)
    pub create_time: Option<DateTime<Utc>>,
}

impl NetworkConnection {
    /// Create a new [`NetworkConnection`] with the info in the form
    ///
    /// # Errors
    ///
    /// * A source, destination, or destination port was not found in the form.
    ///
    /// # Arguments
    ///
    /// * `form` -  The update form
    #[cfg(feature = "api")]
    pub fn from_form(form: super::EntityMetadataForm) -> Result<Self, crate::utils::ApiError> {
        // if we don't have the source field then return an error
        let source = match form.source {
            Some(source) => source,
            None => {
                return crate::bad!("Network Connection entities must have a source!".to_owned());
            }
        };
        // if we don't have the destination field then return an error
        let destination = match form.destination {
            Some(destination) => destination,
            None => {
                return crate::bad!(
                    "Network Connection entities must have a destination!".to_owned()
                );
            }
        };
        // if we don't have the destination_port field then return an error
        let destination_port = match form.destination_port {
            Some(destination_port) => destination_port,
            None => {
                return crate::bad!(
                    "Network Connection entities must have a destination_port!".to_owned()
                );
            }
        };
        // build our network connection entity
        Ok(NetworkConnection {
            protocol: form.protocol,
            source,
            source_port: form.source_port,
            destination,
            destination_port,
            state: form.state,
            pid: form.pid,
            process: form.process,
            create_time: form.create_time,
        })
    }

    /// Add this network connection entity metadata to a form
    ///
    /// # Arguments
    ///
    /// * `form` - The form to add too
    #[cfg(feature = "client")]
    pub fn add_to_form(
        self,
        form: reqwest::multipart::Form,
    ) -> Result<reqwest::multipart::Form, crate::Error> {
        // always set our entity kind
        let form = form
            .text("kind", super::EntityKinds::NetworkConnection.as_str())
            // always set our required fields
            .text("metadata[source]", self.source.to_string())
            .text("metadata[destination_port]", self.destination.to_string())
            .text(
                "metadta[destination_port]",
                self.destination_port.to_string(),
            );
        // set the metadata fields for this entity if htey exist
        let form = multipart_text_to_string!(form, "metadata[protocol]", self.protocol);
        let form = multipart_text_to_string!(form, "metadata[source_port]", self.source_port);
        let form = multipart_text_to_string!(form, "metadata[state]", self.state);
        let form = multipart_text_to_string!(form, "metadata[pid]", self.pid);
        let form = multipart_text!(form, "metadata[process]", self.process.clone());
        let form = multipart_text_to_string!(form, "metadata[create_time]", self.create_time);
        Ok(form)
    }
}
