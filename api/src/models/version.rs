/// The current version of Thorium components
#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Version {
    /// The current version of all Thorium components except the webUI
    #[cfg_attr(feature = "api", schema(value_type = String, example = "1.101.0"))]
    pub thorium: semver::Version,
}

impl Version {
    /// Check if a version is different then our own where the other version is a string
    #[cfg(feature = "client")]
    pub fn compare_thorium(&self, other: &str) -> Result<bool, crate::client::Error> {
        // parse our other version
        let other_cast = semver::Version::parse(other)?;
        // compare our versions
        Ok(other_cast == self.thorium)
    }
}

/// The different operating systems Thorium supports
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum Os {
    Linux,
    Windows,
    Darwin,
}

impl Os {
    /// Get our operating system as an all lowercase str
    pub fn as_str_lowercase(&self) -> &str {
        match self {
            Os::Linux => "linux",
            Os::Windows => "windows",
            Os::Darwin => "darwin",
        }
    }
}

impl std::fmt::Display for Os {
    /// allow os to be displayed
    ///
    /// # Arguments
    ///
    /// * `fmt` - The formatter to write too
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.as_str_lowercase())
    }
}

#[cfg(target_os = "linux")]
impl Default for Os {
    /// set the default os to linux
    fn default() -> Self {
        Os::Linux
    }
}

#[cfg(target_os = "windows")]
impl Default for Os {
    /// set the default os to windows
    fn default() -> Self {
        Os::Windows
    }
}

#[cfg(target_os = "macos")]
impl Default for Os {
    /// set the default os to windows
    fn default() -> Self {
        Os::Darwin
    }
}

/// The different architectures Thorium supports
#[allow(non_camel_case_types)]
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum Arch {
    X86_64,
    AARCH64,
    Arm64,
}

impl Arch {
    /// Get our architecture as an all lowercase str
    pub fn as_str_lowercase(&self) -> &str {
        match self {
            Arch::X86_64 => "x86-64",
            Arch::AARCH64 => "aarch64",
            Arch::Arm64 => "arm64",
        }
    }
}

impl std::fmt::Display for Arch {
    /// allow arch to be displayed
    ///
    /// # Arguments
    ///
    /// * `fmt` - The formatter to write too
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.as_str_lowercase())
    }
}

#[cfg(target_arch = "x86_64")]
impl Default for Arch {
    /// Set a default arch to x86_64 if we are on that platform
    fn default() -> Self {
        Arch::X86_64
    }
}

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
impl Default for Arch {
    // use aarch64 as the default arch
    fn default() -> Self {
        Arch::AARCH64
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
impl Default for Arch {
    /// Set the default arch to arm64 for arm64 macos
    fn default() -> Self {
        Arch::Arm64
    }
}

/// The different components in Thorium that can be auto updated
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum Component {
    /// Thorium's command line tool
    Thorctl,
    /// Thorium's non k8s node controller
    Reactor,
    /// Thorium's agent
    Agent,
}

impl Component {
    /// Get the correct file name with extension for our target os
    ///
    /// # Arguments
    ///
    /// * `os` - The os this component is for
    pub fn to_file_name(&self, os: Os) -> &str {
        match (self, os) {
            (Self::Thorctl, Os::Linux | Os::Darwin) => "thorctl",
            (Self::Thorctl, Os::Windows) => "thorctl.exe",
            //(Self::Thorctl, Os::Darwin) => "thorctl",
            (Self::Reactor, Os::Linux) => "thorium-reactor",
            (Self::Reactor, Os::Windows) => "thorium-reactor.exe",
            (Self::Reactor, Os::Darwin) => "thorium-reactor",
            (Self::Agent, Os::Linux) => "thorium-agent",
            (Self::Agent, Os::Windows) => "thorium-agent.exe",
            (Self::Agent, Os::Darwin) => "thorium-agent",
        }
    }
}
