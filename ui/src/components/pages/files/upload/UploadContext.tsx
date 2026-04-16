import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { useAuth } from '@utilities/auth';
import { useFileUpload } from './useFileUpload';
import { useOriginState } from './useOriginState';
import { handleAssociationUpdate } from './associations';
import { buildUploadFormBase, appendOriginToForm } from './originValidation';
import {
  AssociationCreate,
  AssociationKind,
  CarvedSubType,
  DEFAULT_TLP_SELECTION,
  DropzoneFile,
  FileUploadStatus,
  OriginState,
  OriginType,
  ReactionResultEntry,
  ReactionSubmitResult,
  TagEntry,
  TLPSelection as TLPSelectionState,
} from './types';
import { EntityTypes } from '@models/entities/entities';
import { ReactionSelection } from '@models/reactions';
import { UserInfo } from '@models/users';

interface OriginActions {
  setOriginType: (originType: OriginType) => void;
  setDownloadedField: (field: 'url' | 'name', value: string) => void;
  setParentToolField: (variant: 'transformed' | 'unpacked', field: 'parentFile' | 'tool' | 'toolFlags', value: string) => void;
  setCarvedField: (field: 'parentFile' | 'tool', value: string) => void;
  setCarvedPcapField: (field: 'sourceIp' | 'destinationIp' | 'sourcePort' | 'destinationPort' | 'protocol' | 'url', value: string) => void;
  setCarvedType: (carvedType: CarvedSubType) => void;
  setWireField: (field: 'sniffer' | 'source' | 'destination', value: string) => void;
  setIncidentField: (field: 'incident' | 'coverTerm' | 'missionTeam' | 'network' | 'machine' | 'location', value: string) => void;
  setMemoryDumpField: (field: 'memoryType' | 'parentFile' | 'reconstructed' | 'baseAddress', value: string) => void;
}

interface UploadContextType {
  filesArray: DropzoneFile[];
  setFilesArray: (files: DropzoneFile[]) => void;
  description: string;
  setDescription: (desc: string) => void;
  tags: TagEntry[];
  setTags: (tags: TagEntry[]) => void;
  selectedGroups: string[];
  setSelectedGroups: (groups: string[]) => void;
  selectedTLP: TLPSelectionState;
  handleTLPChange: (newSelection: TLPSelectionState) => void;
  originState: OriginState;
  origin: OriginActions;
  reactionsList: ReactionSelection[];
  setReactionsList: (reactions: ReactionSelection[]) => void;
  userGroups: string[];
  userInfo: UserInfo | null;

  uploadInProgress: boolean;
  activeUploads: string[];
  uploadStatus: Record<string, FileUploadStatus>;
  uploadFailures: Record<string, FormData>;
  uploadStatusDropdown: Record<string, boolean>;
  toggleStatusDropdown: (filePath: string) => void;
  uploadReactionRes: ReactionResultEntry[];
  uploadReactions: Record<string, ReactionSubmitResult[]>;
  uploadReactionFailures: number;
  uploadError: string[];
  uploadSHA256: string[];
  runReactionsRes: ReactionSubmitResult[];
  totalProgress: number;
  showUploadStatus: boolean;

  handleUpload: () => Promise<void>;
  handleBack: () => void;
  cancelUpload: () => void;
  resetStatusMessages: () => void;
  setValidationErrors: (errors: string[]) => void;
  retryFileUpload: (fileName: string) => void;
  retryAllFileUploads: () => void;
  retrySubmitReaction: (status: ReactionResultEntry) => void;
  retryAllReactionSubmissions: () => void;
}

const UploadContext = createContext<UploadContextType | undefined>(undefined);

export const useUpload = (): UploadContextType => {
  const ctx = useContext(UploadContext);
  if (!ctx) throw new Error('useUpload must be used within an UploadProvider');
  return ctx;
};

interface UploadProviderProps {
  entity: EntityTypes | undefined;
  children: React.ReactNode;
}

export const UploadProvider: React.FC<UploadProviderProps> = ({ entity, children }) => {
  const { userInfo } = useAuth();

  const [filesArray, setFilesArray] = useState<DropzoneFile[]>([]);
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState<TagEntry[]>([]);
  const [selectedGroups, setSelectedGroups] = useState<string[]>(entity ? entity.groups : []);
  const [selectedTLP, setSelectedTLP] = useState<TLPSelectionState>({ ...DEFAULT_TLP_SELECTION });
  const [reactionsList, setReactionsList] = useState<ReactionSelection[]>([]);
  const [associations, setAssociations] = useState<AssociationCreate[]>([]);

  const originHook = useOriginState();
  const fileUpload = useFileUpload();

  const userGroups = useMemo(() => (userInfo?.groups ? [...userInfo.groups].sort() : []), [userInfo?.groups]);

  useEffect(() => {
    handleAssociationUpdate([AssociationKind.AssociatedWith], entity, selectedGroups, setAssociations);
  }, [entity, selectedGroups]);

  const tlpTags = (): TagEntry[] => {
    return Object.keys(selectedTLP)
      .filter((tlp) => selectedTLP[tlp])
      .map((tlp) => ({ key: 'TLP', value: tlp }));
  };

  const handleUpload = async () => {
    if (filesArray.length === 0) {
      fileUpload.setValidationErrors(['Please select a file to upload']);
      return;
    }

    const formResult = buildUploadFormBase(description, selectedGroups, tags, tlpTags());
    if ('errors' in formResult) {
      fileUpload.setValidationErrors(formResult.errors);
      return;
    }

    const originResult = appendOriginToForm(formResult.form, originHook.originState);
    if (!originResult.success) {
      fileUpload.setValidationErrors([originResult.error]);
      return;
    }

    await fileUpload.upload(filesArray, formResult.form, selectedGroups, associations, reactionsList);
  };

  const handleTLPChange = (newSelection: TLPSelectionState) => {
    setSelectedTLP(newSelection);
    fileUpload.resetStatusMessages();
  };

  const wrapWithReset = useCallback(
    <TArgs extends unknown[]>(fn: (...args: TArgs) => void) =>
      (...args: TArgs) => {
        fn(...args);
        fileUpload.resetStatusMessages();
      },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const origin = useMemo<OriginActions>(
    () => ({
      setOriginType: wrapWithReset(originHook.setOriginType),
      setDownloadedField: wrapWithReset(originHook.setDownloadedField),
      setParentToolField: wrapWithReset(originHook.setParentToolField),
      setCarvedField: wrapWithReset(originHook.setCarvedField),
      setCarvedPcapField: wrapWithReset(originHook.setCarvedPcapField),
      setCarvedType: wrapWithReset(originHook.setCarvedType),
      setWireField: wrapWithReset(originHook.setWireField),
      setIncidentField: wrapWithReset(originHook.setIncidentField),
      setMemoryDumpField: wrapWithReset(originHook.setMemoryDumpField),
    }),
    [wrapWithReset, originHook],
  );

  const handleBack = () => {
    fileUpload.resetStatusMessages();
    fileUpload.setShowUploadStatus(false);
  };

  const retryFileUpload = (fileName: string) => {
    fileUpload.retryFileUpload(fileName, selectedGroups, associations, reactionsList);
  };

  const retryAllFileUploads = () => {
    fileUpload.retryAllFileUploads(selectedGroups, associations, reactionsList);
  };

  const retrySubmitReaction = (status: ReactionResultEntry) => {
    fileUpload.retrySubmitReaction(status, reactionsList);
  };

  const retryAllReactionSubmissions = () => {
    fileUpload.retryAllReactionSubmissions(reactionsList);
  };

  const value = useMemo<UploadContextType>(
    () => ({
      filesArray,
      setFilesArray,
      description,
      setDescription,
      tags,
      setTags,
      selectedGroups,
      setSelectedGroups,
      selectedTLP,
      handleTLPChange,
      originState: originHook.originState,
      origin,
      reactionsList,
      setReactionsList,
      userGroups,
      userInfo,

      uploadInProgress: fileUpload.uploadInProgress,
      activeUploads: fileUpload.activeUploads,
      uploadStatus: fileUpload.uploadStatus,
      uploadFailures: fileUpload.uploadFailures,
      uploadStatusDropdown: fileUpload.uploadStatusDropdown,
      toggleStatusDropdown: fileUpload.toggleStatusDropdown,
      uploadReactionRes: fileUpload.uploadReactionRes,
      uploadReactions: fileUpload.uploadReactions,
      uploadReactionFailures: fileUpload.uploadReactionFailures,
      uploadError: fileUpload.uploadError,
      uploadSHA256: fileUpload.uploadSHA256,
      runReactionsRes: fileUpload.runReactionsRes,
      totalProgress: fileUpload.totalProgress,
      showUploadStatus: fileUpload.showUploadStatus,

      handleUpload,
      handleBack,
      cancelUpload: fileUpload.cancelUpload,
      resetStatusMessages: fileUpload.resetStatusMessages,
      setValidationErrors: fileUpload.setValidationErrors,
      retryFileUpload,
      retryAllFileUploads,
      retrySubmitReaction,
      retryAllReactionSubmissions,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      filesArray,
      description,
      tags,
      selectedGroups,
      selectedTLP,
      originHook.originState,
      origin,
      reactionsList,
      userGroups,
      userInfo,
      fileUpload.uploadInProgress,
      fileUpload.activeUploads,
      fileUpload.uploadStatus,
      fileUpload.uploadFailures,
      fileUpload.uploadStatusDropdown,
      fileUpload.uploadReactionRes,
      fileUpload.uploadReactions,
      fileUpload.uploadReactionFailures,
      fileUpload.uploadError,
      fileUpload.uploadSHA256,
      fileUpload.runReactionsRes,
      fileUpload.totalProgress,
      fileUpload.showUploadStatus,
    ],
  );

  return <UploadContext.Provider value={value}>{children}</UploadContext.Provider>;
};
