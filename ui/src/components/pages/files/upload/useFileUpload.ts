import { useMemo, useReducer, useRef } from 'react';
import {
  AssociationCreate,
  DEFAULT_UPLOAD_STATE,
  DropzoneFile,
  FileUploadStatus,
  PARALLEL_UPLOAD_LIMIT,
  ReactionResultEntry,
  ReactionSubmitResult,
  UploadAction,
  UploadState,
} from './types';
import { createFileAssociations } from './associations';
import { uploadFile } from '@thorpi/files';
import { SampleSubmissionResponse } from '@models/files';
import { ReactionSelection } from '@models/reactions';
import { submitReactions } from '../reactions/reactions';

function createSemaphore(limit: number) {
  let active = 0;
  const waiting: (() => void)[] = [];

  const acquire = (): Promise<void> => {
    if (active < limit) {
      active++;
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => waiting.push(resolve));
  };

  const release = () => {
    active--;
    const next = waiting.shift();
    if (next) {
      active++;
      next();
    }
  };

  return { acquire, release };
}

function uploadReducer(state: UploadState, action: UploadAction): UploadState {
  switch (action.type) {
    case 'RESET_STATUS':
      return {
        ...state,
        uploadSHA256: [],
        uploadError: [],
        runReactionsRes: [],
        uploadStatus: {},
        uploadReactions: {},
        uploadReactionRes: [],
        uploadReactionFailures: 0,
      };
    case 'SET_UPLOAD_ERROR':
      return { ...state, uploadError: action.errors };
    case 'APPEND_UPLOAD_ERRORS':
      return { ...state, uploadError: [...state.uploadError, ...action.errors] };
    case 'SET_UPLOAD_IN_PROGRESS':
      return { ...state, uploadInProgress: action.value };
    case 'SET_SHOW_UPLOAD_STATUS':
      return { ...state, showUploadStatus: action.value };
    case 'SET_UPLOAD_STATUS':
      return { ...state, uploadStatus: action.status };
    case 'UPDATE_FILE_STATUS':
      return {
        ...state,
        uploadStatus: { ...state.uploadStatus, [action.filePath]: action.status },
      };
    case 'ADD_ACTIVE_UPLOAD':
      return { ...state, activeUploads: [...state.activeUploads, action.filePath] };
    case 'REMOVE_ACTIVE_UPLOAD':
      return {
        ...state,
        activeUploads: state.activeUploads.filter((item) => item !== action.filePath),
      };
    case 'SET_UPLOAD_STATUS_DROPDOWN':
      return { ...state, uploadStatusDropdown: action.dropdown };
    case 'TOGGLE_STATUS_DROPDOWN':
      return {
        ...state,
        uploadStatusDropdown: {
          ...state.uploadStatusDropdown,
          [action.filePath]: !state.uploadStatusDropdown[action.filePath],
        },
      };
    case 'ADD_UPLOAD_FAILURE':
      return {
        ...state,
        uploadFailures: { ...state.uploadFailures, [action.filePath]: action.form },
      };
    case 'REMOVE_UPLOAD_FAILURE': {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const { [action.filePath]: _, ...rest } = state.uploadFailures;
      return { ...state, uploadFailures: rest };
    }
    case 'SET_UPLOAD_SHA256':
      return { ...state, uploadSHA256: action.sha256s };
    case 'SET_UPLOAD_REACTIONS':
      return { ...state, uploadReactions: action.reactions };
    case 'UPDATE_UPLOAD_REACTIONS':
      return {
        ...state,
        uploadReactions: { ...state.uploadReactions, [action.filePath]: action.results },
      };
    case 'UPSERT_REACTION_RESULT': {
      const existingIndex = state.uploadReactionRes.findIndex((r) => r.id === action.entry.id);
      if (existingIndex === -1) {
        return { ...state, uploadReactionRes: [...state.uploadReactionRes, action.entry] };
      }
      return {
        ...state,
        uploadReactionRes: state.uploadReactionRes.map((r, i) => (i === existingIndex ? action.entry : r)),
      };
    }
    case 'ADJUST_REACTION_FAILURES':
      return { ...state, uploadReactionFailures: state.uploadReactionFailures + action.delta };
    case 'SET_TOTAL_UPLOAD_SIZE':
      return { ...state, totalUploadSize: action.size };
    case 'SET_RUN_REACTIONS_RES':
      return { ...state, runReactionsRes: action.results };
    case 'INIT_UPLOAD':
      return {
        ...state,
        uploadStatus: action.filesUploadProgress,
        uploadReactions: action.initReactions,
        uploadReactionRes: [],
        uploadReactionFailures: 0,
        totalUploadSize: action.uploadSize,
        uploadStatusDropdown: action.statusDropdown,
        uploadInProgress: true,
      };
    default:
      return state;
  }
}

export function useFileUpload() {
  const [state, dispatch] = useReducer(uploadReducer, DEFAULT_UPLOAD_STATE);
  const controllerRef = useRef(new AbortController());

  const totalProgress = useMemo(() => {
    if (state.totalUploadSize === 0) return 0;
    let totalUploaded = 0;
    Object.values(state.uploadStatus).forEach((value) => {
      totalUploaded += Math.ceil((value.progress / 100) * value.size);
    });
    return Math.floor((totalUploaded / state.totalUploadSize) * 100);
  }, [state.uploadStatus, state.totalUploadSize]);

  const resetStatusMessages = () => {
    dispatch({ type: 'RESET_STATUS' });
  };

  const setValidationErrors = (errors: string[]) => {
    dispatch({ type: 'SET_UPLOAD_ERROR', errors });
  };

  const setShowUploadStatus = (value: boolean) => {
    dispatch({ type: 'SET_SHOW_UPLOAD_STATUS', value });
  };

  const toggleStatusDropdown = (filePath: string) => {
    dispatch({ type: 'TOGGLE_STATUS_DROPDOWN', filePath });
  };

  const trackAndSubmitReactions = (
    sha256: string,
    submission: { path: string; size: number },
    submitReactionsList: ReactionSelection[],
  ) => {
    const allRunReactionsRes: ReactionSubmitResult[] = [];
    return submitReactions(sha256, submitReactionsList).then((submitRes: ReactionSubmitResult[]) => {
      let error = false;
      submitRes.forEach((value: ReactionSubmitResult, index: number) => {
        if (value.error !== '') {
          error = true;
        } else {
          dispatch({ type: 'ADJUST_REACTION_FAILURES', delta: -1 });
        }
        dispatch({
          type: 'UPSERT_REACTION_RESULT',
          entry: {
            id: sha256 + value.pipeline,
            sha256,
            result: value,
            submission,
          },
        });
        submitRes[index] = {
          ...submitRes[index],
          path: submission.path,
          size: submission.size,
          sha256,
        };
      });
      dispatch({ type: 'UPDATE_UPLOAD_REACTIONS', filePath: submission.path, results: submitRes });
      allRunReactionsRes.push(...submitRes);
      if (error) {
        dispatch({
          type: 'UPDATE_FILE_STATUS',
          filePath: submission.path,
          status: {
            progress: 100,
            size: submission.size,
            type: 'warning',
            msg: 'Error submitting reactions',
            sha256,
            fileFail: false,
            reactionFail: true,
          },
        });
      } else {
        dispatch({
          type: 'UPDATE_FILE_STATUS',
          filePath: submission.path,
          status: {
            progress: 100,
            size: submission.size,
            type: 'success',
            msg: 'Upload successful!',
            sha256,
            fileFail: false,
            reactionFail: false,
          },
        });
      }
      dispatch({ type: 'SET_RUN_REACTIONS_RES', results: [...allRunReactionsRes] });
    });
  };

  const trackAndUploadFile = (
    form: FormData,
    selectedGroups: string[],
    associations: AssociationCreate[],
    reactionsList: ReactionSelection[],
  ) => {
    const allResSha256: string[] = [];
    const allResErrors: string[] = [];
    const submission = form.get('data') as DropzoneFile;
    const filePath = submission.path || submission.name;
    dispatch({ type: 'ADD_ACTIVE_UPLOAD', filePath });

    const uploadFileProgressHandler = (progress: number) => {
      if (progress < 1) {
        dispatch({
          type: 'UPDATE_FILE_STATUS',
          filePath,
          status: {
            progress: Math.floor(progress * 100),
            size: submission.size,
            type: 'info',
            msg: 'Upload in progress',
            sha256: '',
            fileFail: false,
            reactionFail: false,
          },
        });
      }
    };

    const addUploadErrorMsg = (error: string) => {
      allResErrors.push(error);
      dispatch({
        type: 'UPDATE_FILE_STATUS',
        filePath,
        status: {
          progress: 100,
          size: submission.size,
          type: 'danger',
          msg: error,
          sha256: '',
          fileFail: true,
          reactionFail: true,
        },
      });
    };

    return uploadFile(form, addUploadErrorMsg, uploadFileProgressHandler, controllerRef.current).then(
      async (response: SampleSubmissionResponse | false) => {
        if (response) {
          allResSha256.push(response.sha256);
          dispatch({
            type: 'UPDATE_FILE_STATUS',
            filePath,
            status: {
              progress: 99,
              size: submission.size,
              type: 'info',
              msg: 'Submitting reactions',
              sha256: response.sha256,
              fileFail: false,
              reactionFail: false,
            },
          });
          await createFileAssociations(response.sha256, selectedGroups, associations);
          dispatch({ type: 'ADJUST_REACTION_FAILURES', delta: reactionsList.length });
          await trackAndSubmitReactions(response.sha256, { path: filePath, size: submission.size }, reactionsList);
          dispatch({ type: 'REMOVE_UPLOAD_FAILURE', filePath });
        } else {
          dispatch({ type: 'ADD_UPLOAD_FAILURE', filePath, form });
        }
        dispatch({ type: 'SET_UPLOAD_SHA256', sha256s: [...allResSha256] });
        if (allResErrors.length > 0) {
          dispatch({ type: 'APPEND_UPLOAD_ERRORS', errors: allResErrors });
        }
        dispatch({ type: 'REMOVE_ACTIVE_UPLOAD', filePath });
      },
    );
  };

  const upload = async (
    filesArray: DropzoneFile[],
    formBase: FormData,
    selectedGroups: string[],
    associations: AssociationCreate[],
    reactionsList: ReactionSelection[],
  ) => {
    if (filesArray.length > 1) {
      dispatch({ type: 'SET_SHOW_UPLOAD_STATUS', value: true });
    }

    const filesUploadProgress: Record<string, FileUploadStatus> = {};
    const statusDropdown: Record<string, boolean> = {};
    const initReactions: Record<string, ReactionSubmitResult[]> = {};
    let uploadSize = 0;

    for (const submission of filesArray) {
      const filePath = submission.path || submission.name;
      if (submission) {
        uploadSize += submission.size;
        filesUploadProgress[filePath] = {
          progress: 0,
          size: submission.size,
          type: 'info',
          msg: 'Queued',
          sha256: '',
          fileFail: false,
          reactionFail: false,
        };
      }
      statusDropdown[filePath] = false;
      initReactions[filePath] = [];
    }

    dispatch({
      type: 'INIT_UPLOAD',
      filesUploadProgress,
      statusDropdown,
      uploadSize,
      initReactions,
    });

    const sem = createSemaphore(PARALLEL_UPLOAD_LIMIT);
    const uploadPromises: Promise<void>[] = [];
    for (const submission of filesArray) {
      const newForm = new FormData();
      for (const [key, val] of formBase.entries()) {
        newForm.append(key, val);
      }
      if (submission) {
        newForm.set('data', submission);
      }

      uploadPromises.push(
        sem.acquire().then(async () => {
          try {
            await trackAndUploadFile(newForm, selectedGroups, associations, reactionsList);
          } finally {
            sem.release();
          }
        }),
      );
    }
    await Promise.all(uploadPromises);
    dispatch({ type: 'SET_UPLOAD_IN_PROGRESS', value: false });
  };

  const retryFileUpload = async (
    fileName: string,
    selectedGroups: string[],
    associations: AssociationCreate[],
    reactionsList: ReactionSelection[],
  ) => {
    dispatch({ type: 'SET_UPLOAD_IN_PROGRESS', value: true });
    await trackAndUploadFile(state.uploadFailures[fileName], selectedGroups, associations, reactionsList);
    dispatch({ type: 'SET_UPLOAD_IN_PROGRESS', value: false });
  };

  const retryAllFileUploads = async (selectedGroups: string[], associations: AssociationCreate[], reactionsList: ReactionSelection[]) => {
    dispatch({ type: 'SET_UPLOAD_IN_PROGRESS', value: true });
    const failureSnapshot = { ...state.uploadFailures };

    const sem = createSemaphore(PARALLEL_UPLOAD_LIMIT);
    const uploadPromises: Promise<void>[] = [];
    for (const form of Object.values(failureSnapshot)) {
      uploadPromises.push(
        sem.acquire().then(async () => {
          try {
            await trackAndUploadFile(form, selectedGroups, associations, reactionsList);
          } finally {
            sem.release();
          }
        }),
      );
    }
    await Promise.all(uploadPromises);
    dispatch({ type: 'SET_UPLOAD_IN_PROGRESS', value: false });
  };

  const retrySubmitReaction = async (status: ReactionResultEntry, reactionsList: ReactionSelection[]) => {
    dispatch({ type: 'SET_UPLOAD_IN_PROGRESS', value: true });
    dispatch({
      type: 'UPDATE_FILE_STATUS',
      filePath: status.submission.path,
      status: {
        progress: 99,
        size: status.submission.size,
        type: 'info',
        msg: 'Submitting reactions',
        sha256: status.sha256,
        fileFail: false,
        reactionFail: false,
      },
    });
    const failedReaction = state.uploadReactionRes.filter((failure) => failure.id === status.id)[0];
    await trackAndSubmitReactions(
      status.sha256,
      failedReaction.submission,
      reactionsList.filter((reaction: ReactionSelection) => reaction.pipeline === failedReaction.result.pipeline),
    );
    dispatch({ type: 'SET_UPLOAD_IN_PROGRESS', value: false });
  };

  const retryAllReactionSubmissions = async (reactionsList: ReactionSelection[]) => {
    dispatch({ type: 'SET_UPLOAD_IN_PROGRESS', value: true });
    const submissionPromises: Promise<void>[] = [];
    state.uploadReactionRes.forEach((value) => {
      if (value.result.error !== '') {
        const failedReaction = state.uploadReactionRes.filter((failure) => failure.id === value.id)[0];
        submissionPromises.push(
          trackAndSubmitReactions(
            value.sha256,
            failedReaction.submission,
            reactionsList.filter((reaction: ReactionSelection) => reaction.pipeline === failedReaction.result.pipeline),
          ),
        );
      }
    });
    await Promise.all(submissionPromises);
    dispatch({ type: 'SET_UPLOAD_IN_PROGRESS', value: false });
  };

  const cancelUpload = () => {
    controllerRef.current.abort();
    controllerRef.current = new AbortController();
  };

  return {
    uploadError: state.uploadError,
    runReactionsRes: state.runReactionsRes,
    uploadSHA256: state.uploadSHA256,
    uploadInProgress: state.uploadInProgress,
    activeUploads: state.activeUploads,
    uploadStatus: state.uploadStatus,
    uploadFailures: state.uploadFailures,
    uploadStatusDropdown: state.uploadStatusDropdown,
    toggleStatusDropdown,
    uploadReactionRes: state.uploadReactionRes,
    uploadReactions: state.uploadReactions,
    uploadReactionFailures: state.uploadReactionFailures,
    totalUploadSize: state.totalUploadSize,
    showUploadStatus: state.showUploadStatus,
    totalProgress,
    setValidationErrors,
    setShowUploadStatus,
    resetStatusMessages,
    upload,
    retryFileUpload,
    retryAllFileUploads,
    retrySubmitReaction,
    retryAllReactionSubmissions,
    cancelUpload,
  };
}
