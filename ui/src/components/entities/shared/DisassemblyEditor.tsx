import { useEffect, useRef, useState } from 'react';

// project imports
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import CodeEditor from '@components/shared/inputs/code/CodeEditor/CodeEditor';
import { disassemblyToText, textToDisassembly } from '@utilities/disassembly';
import { FormatType } from '@utilities/rules/types';
import { CompiledInstruction } from '@models/entities/functions';

/** Props for {@link DisassemblyEditor}. */
interface DisassemblyEditorProps {
  /** The structured disassembly to seed the editor buffer from. */
  disassembly: CompiledInstruction[];
  /** Called with the parsed instructions whenever the buffer parses cleanly. */
  onCommit: (instructions: CompiledInstruction[]) => void;
  /**
   * When provided, changing this reference re-seeds the editor buffer from `disassembly` and clears
   * any parse error. Used by the details view to re-seed from the canonical entity once it loads or is
   * saved; the create view omits it so the buffer is never reset from its own committed edits.
   */
  resetSignal?: unknown;
}

/**
 * Text editor for a compiled function's disassembly.
 *
 * Disassembly is edited as address-per-line text but stored as structured instructions, so the buffer
 * is kept locally and only committed via `onCommit` when it parses cleanly; malformed input surfaces an
 * inline error and leaves the last committed value untouched.
 */
const DisassemblyEditor = ({ disassembly, onCommit, resetSignal }: DisassemblyEditorProps) => {
  const [disassemblyText, setDisassemblyText] = useState(() => disassemblyToText(disassembly));
  const [parseError, setParseError] = useState<string | undefined>();
  // hold the latest disassembly in a ref so the re-seed effect can read it without depending on it
  // (depending on it would re-seed on every committed keystroke and clobber in-progress edits)
  const disassemblyRef = useRef(disassembly);
  disassemblyRef.current = disassembly;
  // re-seed the buffer from the canonical disassembly whenever the reset signal changes; this only
  // fires on load/save (not while typing), so it never clobbers in-progress edits
  useEffect(() => {
    if (resetSignal === undefined) {
      return;
    }
    setDisassemblyText(disassemblyToText(disassemblyRef.current));
    setParseError(undefined);
  }, [resetSignal]);
  const handleChange = (text: string): void => {
    setDisassemblyText(text);
    const { instructions, error } = textToDisassembly(text);
    setParseError(error);
    // only commit valid disassembly; malformed input leaves the last good value in place
    if (!error) {
      onCommit(instructions);
    }
  };
  return (
    <>
      <CodeEditor value={disassemblyText} onChange={handleChange} format={FormatType.Disassembly} height="400px" />
      {parseError && (
        <AlertBanner className="mt-2" severity={Severity.Error}>
          {parseError}
        </AlertBanner>
      )}
    </>
  );
};

export default DisassemblyEditor;
