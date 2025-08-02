import React, { useMemo } from 'react';
import { useDropzone } from 'react-dropzone';

// project imports
import { Subtitle } from '@components';

interface DropzoneProperties {
  width: string; // width of dropzone area
  onChange: (files: any[]) => void;
  onError?: (errors: string[]) => void;
  selectedFiles: any[];
}

export const UploadDropzone: React.FC<DropzoneProperties> = ({ width, onChange, onError, selectedFiles }) => {
  const baseStyle = {
    flex: 1,
    display: 'flex',
    flexDirection: 'column' as 'column',
    alignItems: 'center',
    padding: '20px',
    borderWidth: 2,
    borderRadius: 2,
    borderColor: 'var(--thorium-panel-border-color)',
    borderStyle: 'dashed',
    backgroundColor: 'var(--thorium-panel-color)',
    color: 'var(--thorium-text-color)',
    outline: 'none',
    transition: 'border .24s ease-in-out',
    width: width,
  };

  const activeStyle = {
    borderColor: '#2196f3',
  };

  const acceptStyle = {
    borderColor: '#00e676',
  };

  const rejectStyle = {
    borderColor: '#ff1744',
  };

  // Configured drop zone including max # of files and their max size
  const { acceptedFiles, fileRejections, getRootProps, getInputProps, isDragActive, isDragAccept, isDragReject } = useDropzone({
    maxSize: 10737418240,
    onDrop: (acceptedFiles) => {
      onChange(acceptedFiles);
      if (onError) onError([]);
    },
  });

  const style = useMemo(
    () => ({
      ...baseStyle,
      ...(isDragActive ? activeStyle : {}),
      ...(isDragAccept ? acceptStyle : {}),
      ...(isDragReject ? rejectStyle : {}),
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }),
    [isDragActive, isDragReject, isDragAccept],
  );

  // files accepted and their size
  let fileAcceptedItems: React.JSX.Element[] = [];
  if (acceptedFiles && acceptedFiles.length > 0) {
    fileAcceptedItems = acceptedFiles.map((file) => (
      <li key={file.path}>
        {file.path} - {file.size} bytes
      </li>
    ));
  } else if (selectedFiles && selectedFiles.length > 0) {
    fileAcceptedItems = selectedFiles.map((file) => (
      <li key={file.path}>
        {file.path} - {file.size} bytes
      </li>
    ));
  }

  // file rejection list
  const fileRejectionItems = fileRejections.map(({ file, errors }) => {
    return (
      <li key={file.path}>
        {file.path} - {file.size} bytes
        <ul>
          {errors.map((e) => (
            <li key={e.code}>{e.message}</li>
          ))}
        </ul>
      </li>
    );
  });

  return (
    <div>
      <div {...getRootProps({ style })}>
        <input {...getInputProps()} />
        <p>Drag and drop some files or click to select files</p>
        <em>({`files must be < 10GB`})</em>
      </div>
      <aside>
        {(fileAcceptedItems.length > 0 || fileRejectionItems.length > 0) && <hr />}
        {fileAcceptedItems.length > 0 && (
          <>
            <h4>
              <Subtitle>Accepted Files</Subtitle>
            </h4>
            <ul>
              <Subtitle>{fileAcceptedItems}</Subtitle>
            </ul>
          </>
        )}
        {fileRejectionItems.length > 0 && (
          <>
            <h4>
              <Subtitle>Rejected Files</Subtitle>
            </h4>
            <ul>
              <Subtitle>{fileRejectionItems}</Subtitle>
            </ul>
          </>
        )}
      </aside>
    </div>
  );
};
