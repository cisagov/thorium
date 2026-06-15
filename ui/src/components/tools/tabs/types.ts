// project imports
import { Output, OutputDisplayType } from '@models/results';

/** The tabs available within a tool-result tile. */
export enum ToolResultTabKey {
  Result = 'result',
  Files = 'files',
  Children = 'children',
  Entities = 'entities',
}

/** Shared props passed to each tool-result tab body. */
export interface ToolResultTabProps {
  result: Output;
  sha256: string;
  tool: string;
  type: OutputDisplayType;
}
