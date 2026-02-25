// project imports
import { Output } from '@models/results';

export type ResultRenderProps = {
  result: Output;
  sha256: string;
  tool: string;
};
