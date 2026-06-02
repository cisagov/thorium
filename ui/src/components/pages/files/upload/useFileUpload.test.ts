import { describe, it, expect } from 'vitest';

// project imports
import { uploadReducer, createSemaphore } from './useFileUpload';
import { DEFAULT_UPLOAD_STATE, FileUploadStatus, UploadState } from './types';

function stateWith(patch: Partial<UploadState>): UploadState {
  return { ...structuredClone(DEFAULT_UPLOAD_STATE), ...patch };
}

const sampleStatus: FileUploadStatus = {
  progress: 50,
  size: 1000,
  type: 'info',
  msg: 'Upload in progress',
  sha256: '',
  fileFail: false,
  reactionFail: false,
};

describe('uploadReducer', () => {
  it('RESET_STATUS clears all status fields', () => {
    const state = stateWith({
      uploadSHA256: ['abc'],
      uploadError: ['err'],
      runReactionsRes: [{ error: '', group: 'g', pipeline: 'p' }],
      uploadStatus: { 'file.bin': sampleStatus },
      uploadReactions: { 'file.bin': [] },
      uploadReactionRes: [],
      uploadReactionFailures: 3,
    });
    const next = uploadReducer(state, { type: 'RESET_STATUS' });
    expect(next.uploadSHA256).toEqual([]);
    expect(next.uploadError).toEqual([]);
    expect(next.runReactionsRes).toEqual([]);
    expect(next.uploadStatus).toEqual({});
    expect(next.uploadReactions).toEqual({});
    expect(next.uploadReactionRes).toEqual([]);
    expect(next.uploadReactionFailures).toBe(0);
  });

  it('SET_UPLOAD_ERROR sets errors', () => {
    const next = uploadReducer(DEFAULT_UPLOAD_STATE, {
      type: 'SET_UPLOAD_ERROR',
      errors: ['err1', 'err2'],
    });
    expect(next.uploadError).toEqual(['err1', 'err2']);
  });

  it('APPEND_UPLOAD_ERRORS appends to existing', () => {
    const state = stateWith({ uploadError: ['existing'] });
    const next = uploadReducer(state, {
      type: 'APPEND_UPLOAD_ERRORS',
      errors: ['new1', 'new2'],
    });
    expect(next.uploadError).toEqual(['existing', 'new1', 'new2']);
  });

  it('SET_UPLOAD_IN_PROGRESS toggles flag', () => {
    const next = uploadReducer(DEFAULT_UPLOAD_STATE, {
      type: 'SET_UPLOAD_IN_PROGRESS',
      value: true,
    });
    expect(next.uploadInProgress).toBe(true);
  });

  it('SET_SHOW_UPLOAD_STATUS toggles flag', () => {
    const next = uploadReducer(DEFAULT_UPLOAD_STATE, {
      type: 'SET_SHOW_UPLOAD_STATUS',
      value: true,
    });
    expect(next.showUploadStatus).toBe(true);
  });

  it('SET_UPLOAD_STATUS replaces entire status map', () => {
    const status = { 'f1.bin': sampleStatus };
    const next = uploadReducer(DEFAULT_UPLOAD_STATE, {
      type: 'SET_UPLOAD_STATUS',
      status,
    });
    expect(next.uploadStatus).toEqual(status);
  });

  it('UPDATE_FILE_STATUS updates specific file', () => {
    const state = stateWith({
      uploadStatus: { 'f1.bin': sampleStatus, 'f2.bin': sampleStatus },
    });
    const updated: FileUploadStatus = { ...sampleStatus, progress: 100, type: 'success', msg: 'Done' };
    const next = uploadReducer(state, {
      type: 'UPDATE_FILE_STATUS',
      filePath: 'f1.bin',
      status: updated,
    });
    expect(next.uploadStatus['f1.bin'].progress).toBe(100);
    expect(next.uploadStatus['f2.bin'].progress).toBe(50);
  });

  it('ADD_ACTIVE_UPLOAD adds to list', () => {
    const next = uploadReducer(DEFAULT_UPLOAD_STATE, {
      type: 'ADD_ACTIVE_UPLOAD',
      filePath: 'file.bin',
    });
    expect(next.activeUploads).toEqual(['file.bin']);
  });

  it('REMOVE_ACTIVE_UPLOAD removes from list', () => {
    const state = stateWith({ activeUploads: ['a.bin', 'b.bin', 'c.bin'] });
    const next = uploadReducer(state, {
      type: 'REMOVE_ACTIVE_UPLOAD',
      filePath: 'b.bin',
    });
    expect(next.activeUploads).toEqual(['a.bin', 'c.bin']);
  });

  it('TOGGLE_STATUS_DROPDOWN toggles specific file', () => {
    const state = stateWith({
      uploadStatusDropdown: { 'a.bin': false, 'b.bin': true },
    });
    const next = uploadReducer(state, {
      type: 'TOGGLE_STATUS_DROPDOWN',
      filePath: 'a.bin',
    });
    expect(next.uploadStatusDropdown['a.bin']).toBe(true);
    expect(next.uploadStatusDropdown['b.bin']).toBe(true);

    const next2 = uploadReducer(next, {
      type: 'TOGGLE_STATUS_DROPDOWN',
      filePath: 'b.bin',
    });
    expect(next2.uploadStatusDropdown['b.bin']).toBe(false);
  });

  it('ADD_UPLOAD_FAILURE adds failure entry', () => {
    const form = new FormData();
    const next = uploadReducer(DEFAULT_UPLOAD_STATE, {
      type: 'ADD_UPLOAD_FAILURE',
      filePath: 'fail.bin',
      form,
    });
    expect(next.uploadFailures['fail.bin']).toBe(form);
  });

  it('REMOVE_UPLOAD_FAILURE removes failure entry', () => {
    const form1 = new FormData();
    const form2 = new FormData();
    const state = stateWith({
      uploadFailures: { 'a.bin': form1, 'b.bin': form2 },
    });
    const next = uploadReducer(state, {
      type: 'REMOVE_UPLOAD_FAILURE',
      filePath: 'a.bin',
    });
    expect(next.uploadFailures).toEqual({ 'b.bin': form2 });
  });

  it('SET_UPLOAD_SHA256 sets sha256 list', () => {
    const next = uploadReducer(DEFAULT_UPLOAD_STATE, {
      type: 'SET_UPLOAD_SHA256',
      sha256s: ['abc', 'def'],
    });
    expect(next.uploadSHA256).toEqual(['abc', 'def']);
  });

  it('UPSERT_REACTION_RESULT inserts new entry', () => {
    const entry = {
      id: 'sha256-pipe1',
      sha256: 'abc',
      result: { error: '', group: 'g', pipeline: 'pipe1' },
      submission: { path: 'f.bin', size: 100 },
    };
    const next = uploadReducer(DEFAULT_UPLOAD_STATE, {
      type: 'UPSERT_REACTION_RESULT',
      entry,
    });
    expect(next.uploadReactionRes).toHaveLength(1);
    expect(next.uploadReactionRes[0].id).toBe('sha256-pipe1');
  });

  it('UPSERT_REACTION_RESULT updates existing entry', () => {
    const entry1 = {
      id: 'sha256-pipe1',
      sha256: 'abc',
      result: { error: 'fail', group: 'g', pipeline: 'pipe1' },
      submission: { path: 'f.bin', size: 100 },
    };
    const state = stateWith({ uploadReactionRes: [entry1] });
    const entry2 = { ...entry1, result: { ...entry1.result, error: '' } };
    const next = uploadReducer(state, {
      type: 'UPSERT_REACTION_RESULT',
      entry: entry2,
    });
    expect(next.uploadReactionRes).toHaveLength(1);
    expect(next.uploadReactionRes[0].result.error).toBe('');
  });

  it('ADJUST_REACTION_FAILURES adjusts count', () => {
    const state = stateWith({ uploadReactionFailures: 3 });
    const next = uploadReducer(state, {
      type: 'ADJUST_REACTION_FAILURES',
      delta: -1,
    });
    expect(next.uploadReactionFailures).toBe(2);
  });

  it('SET_TOTAL_UPLOAD_SIZE sets size', () => {
    const next = uploadReducer(DEFAULT_UPLOAD_STATE, {
      type: 'SET_TOTAL_UPLOAD_SIZE',
      size: 5000,
    });
    expect(next.totalUploadSize).toBe(5000);
  });

  it('INIT_UPLOAD sets initial upload state', () => {
    const filesUploadProgress = {
      'a.bin': { ...sampleStatus, progress: 0, msg: 'Queued' },
      'b.bin': { ...sampleStatus, progress: 0, msg: 'Queued' },
    };
    const next = uploadReducer(DEFAULT_UPLOAD_STATE, {
      type: 'INIT_UPLOAD',
      filesUploadProgress,
      statusDropdown: { 'a.bin': false, 'b.bin': false },
      uploadSize: 2000,
      initReactions: { 'a.bin': [], 'b.bin': [] },
    });
    expect(next.uploadInProgress).toBe(true);
    expect(next.totalUploadSize).toBe(2000);
    expect(Object.keys(next.uploadStatus)).toHaveLength(2);
    expect(next.uploadReactionRes).toEqual([]);
    expect(next.uploadReactionFailures).toBe(0);
  });

  it('returns state unchanged for unknown action', () => {
    const state = structuredClone(DEFAULT_UPLOAD_STATE);
    // @ts-expect-error testing unknown action
    const next = uploadReducer(state, { type: 'UNKNOWN_ACTION' });
    expect(next).toEqual(state);
  });
});

describe('createSemaphore', () => {
  it('allows up to limit concurrent acquires', async () => {
    const sem = createSemaphore(2);
    await sem.acquire();
    await sem.acquire();
    let resolved = false;
    const p = sem.acquire().then(() => {
      resolved = true;
    });
    await Promise.resolve();
    expect(resolved).toBe(false);
    sem.release();
    await p;
    expect(resolved).toBe(true);
  });

  it('processes waiting in FIFO order', async () => {
    const sem = createSemaphore(1);
    await sem.acquire();
    const order: number[] = [];
    const p1 = sem.acquire().then(() => {
      order.push(1);
    });
    const p2 = sem.acquire().then(() => {
      order.push(2);
    });
    sem.release();
    await p1;
    sem.release();
    await p2;
    expect(order).toEqual([1, 2]);
  });

  it('handles rapid acquire/release cycles', async () => {
    const sem = createSemaphore(3);
    const results: number[] = [];
    const tasks = Array.from({ length: 10 }, (_, i) =>
      sem.acquire().then(() => {
        results.push(i);
        sem.release();
      }),
    );
    await Promise.all(tasks);
    expect(results).toHaveLength(10);
  });
});
