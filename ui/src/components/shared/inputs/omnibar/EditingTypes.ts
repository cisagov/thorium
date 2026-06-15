import { Clause, ClauseDraft, ClausePart } from './ClauseTypes';

// spec: ./SPEC.md

export enum KeyName {
  Enter = 'Enter',
  Tab = 'Tab',
  ArrowDown = 'ArrowDown',
  ArrowUp = 'ArrowUp',
  ArrowLeft = 'ArrowLeft',
  ArrowRight = 'ArrowRight',
}

export enum OmnibarEditMode {
  Idle = 'idle',
  EditExisting = 'edit-existing',
  EditNew = 'edit-new',
}

export type EditSession =
  | { mode: OmnibarEditMode.Idle }
  | { mode: OmnibarEditMode.EditExisting; part: ClausePart; textDraft: string; clauseDraft: Clause; clauseIdx: number }
  | { mode: OmnibarEditMode.EditNew; part: ClausePart; textDraft: string; clauseDraft: ClauseDraft };

export type DropdownOption = {
  value: string;
  category: string;
  helpText?: string;
  hasCheckmark?: boolean;
};

export type DropdownState = {
  index: number;
  isSelecting: boolean;
};

export function NewEditSessionClean(): EditSession {
  return { mode: OmnibarEditMode.EditNew, part: 'category', textDraft: '', clauseDraft: {} };
}
