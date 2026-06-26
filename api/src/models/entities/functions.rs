//! A function in a binary

/// A single disassembled instruction from a function
#[derive(Debug, Clone, Default, Serialize, Deserialize, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct CompiledInstruction {
    /// The address this instruction starts at
    pub address: u64,
    /// The contents of this instruction
    pub instruction: String,
}

/// A compiled function
#[derive(Debug, Clone, Default, Serialize, Deserialize, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct CompiledFunction {
    /// The address this function is located at
    pub address: u64,
    /// The dissasembled instructions for this function
    pub disassembly: Vec<CompiledInstruction>,
}

impl CompiledFunction {
    /// Add this compiled functions entity metadata to a form
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
        let mut form = form
            .text("kind", super::EntityKinds::CompiledFunction.as_str())
            .text("metadata[function_address]", self.address.to_string());
        // add this functions metadata (list fields require a trailing `[]`)
        crate::multipart_list_serialize!(form, "metadata[disassembly][]", self.disassembly);
        Ok(form)
    }

    /// Create a new compiled function entity with the info in the form
    ///
    /// # Arguments
    ///
    /// * `form` -  The update form
    #[cfg(feature = "api")]
    pub fn from_form(form: super::EntityMetadataForm) -> Result<Self, crate::utils::ApiError> {
        // if we don't have the address set then return an error
        let address = match form.function_address {
            Some(address) => address,
            None => {
                return crate::bad!("Compiled function entities must have a address!".to_owned());
            }
        };
        // build incident entity
        Ok(CompiledFunction {
            address,
            disassembly: form.disassembly,
        })
    }
}

/// The decompilation for a function
#[derive(Debug, Clone, Default, Serialize, Deserialize, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct DecompiledFunction {
    /// The address this descompiled function is located at
    pub address: u64,
    /// The tools that decompiled this function
    pub tools: Vec<String>,
    /// The decompiled function
    pub content: String,
}

impl DecompiledFunction {
    /// Add this decompiled functions entity metadata to a form
    ///
    /// # Arguments
    ///
    /// * `form` - The form to add too
    #[cfg(feature = "client")]
    pub fn add_to_form(
        mut self,
        form: reqwest::multipart::Form,
    ) -> Result<reqwest::multipart::Form, crate::Error> {
        // always set our entity kind
        let form = form
            .text("kind", super::EntityKinds::DecompiledFunction.as_str())
            .text("metadata[function_address]", self.address.to_string())
            .text("metadata[decompilation_content]", self.content);
        // add this functions tools (plain strings; list fields require a trailing `[]`)
        let form = crate::multipart_list!(form, "metadata[tools][]", self.tools);
        Ok(form)
    }

    /// Create a new compiled function entity with the info in the form
    ///
    /// # Arguments
    ///
    /// * `form` -  The update form
    #[cfg(feature = "api")]
    pub fn from_form(form: super::EntityMetadataForm) -> Result<Self, crate::utils::ApiError> {
        // if we don't have the address set then return an error
        let address = match form.function_address {
            Some(address) => address,
            None => {
                return crate::bad!("Decompiled function entities must have a address!".to_owned());
            }
        };
        // if we don't have any decompilation content set then return an error
        let content = match form.decompilation_content {
            Some(content) => content,
            None => {
                return crate::bad!(
                    "Decompiled function entities must have decompilation content!".to_owned()
                );
            }
        };
        Ok(DecompiledFunction {
            address,
            tools: form.tools,
            content,
        })
    }
}
