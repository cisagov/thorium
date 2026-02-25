// project imports
import { ImageVersion } from './images';

/// Represents any valid JSON value.
export type Value = null | boolean | number | string | Value[] | { [key: string]: Value } | {};

/// The type of display class to use in the UI for this output
export enum OutputDisplayType {
  /// Render this output as json
  Json = 'JSON',
  /// Render this output as a string
  String = 'String',
  /// Render this output as a table
  Table = 'Table',
  /// Render this output as one or more images
  Image = 'Image',
  /// Use a custom render class in the UI, class will be based on tool name
  Custom = 'Custom',
  /// Result to render is disassembly
  Disassembly = 'Disassembly',
  /// Result to render is HTML and may need sanitization before rendering
  Html = 'HTML',
  /// Result to render is Markdown formatted
  Markdown = 'Markdown',
  /// Do not return this output unless its requested
  Hidden = 'Hidden',
  /// Result to render is XML formatted
  Xml = 'XML',
}

/// A single result for a single run of a tool with a specific command
export type Output = {
  /// The id for this result
  id: string;
  /// The groups that can see this result
  groups: string[];
  /// The version of the tool that generated this result
  tool_version?: ImageVersion;
  /// The command used to generate this result
  cmd?: string;
  /// When this result was uploaded
  uploaded: string;
  /// Set to true if a deserialization failure occurred
  deserialization_error?: string;
  /// The result
  result: Value;
  /// Any files tied to this result
  files?: string[];
  /// The display type of this tool output
  display_type: OutputDisplayType;
  /// The children that were found when generating this result
  children: { [child: string]: string };
};
