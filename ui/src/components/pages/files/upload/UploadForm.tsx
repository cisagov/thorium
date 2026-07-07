import React, { ReactNode } from 'react';
import { Button, Col, Form, Row } from 'react-bootstrap';
import Subtitle from '@components/shared/titles/Subtitle';
import UploadDropzone from '@components/shared/UploadDropzone';
import { TagSelect } from '@components/shared/inputs/tags/TagSelect';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import SelectPipelines from '../reactions/SelectPipelines';
import ProgressBarContainer from './ProgressBarContainer';
import TLPSelection from './TLPSelection';
import OriginForm from './OriginForm';
import UploadAlertBanner from './UploadAlertBanner';
import { useUpload } from './UploadContext';

type UploadFieldProps = {
  label: string;
  marker?: '*' | 'T';
  disabled?: boolean;
  rowClassName?: string;
  children: ReactNode;
}

const UploadField: React.FC<UploadFieldProps> = ({
  label,
  marker,
  disabled = false,
  rowClassName = 'mb-4',
  children,
}) => {

  const title = (
    <Subtitle>
      {label} {marker && <sup>{marker}</sup>}
    </Subtitle>
  )

  return (
    <>
      <Row className={rowClassName}>
        <Col xs={12} xl={2} className="upload-field-name">
          {title}
        </Col>
        <Col xs={12} xl={10} className={`${disabled ? 'disabled ' : ''}upload-field`}>
          {children}
        </Col>
      </Row>
    </>
  )
}

const UploadForm: React.FC = () => {
  const {
    uploadInProgress,
    filesArray,
    setFilesArray,
    selectedGroups,
    setSelectedGroups,
    userGroups,
    description,
    setDescription,
    tags,
    setTags,
    selectedTLP,
    handleTLPChange,
    reactionsList,
    setReactionsList,
    userInfo,
    handleUpload,
    uploadStatus,
    uploadError,
    setValidationErrors,
    uploadSHA256,
    runReactionsRes,
    resetStatusMessages,
  } = useUpload();

  return (
    <>
      <UploadField
        label='File'
        marker='*'
        disabled={uploadInProgress}
      >
        <UploadDropzone onChange={setFilesArray} onError={setValidationErrors} selectedFiles={filesArray} width="100%" />
      </UploadField>

      <UploadField
        label='Groups'
        marker='*'
        disabled={uploadInProgress}
        rowClassName='mb-4'
      >
        <SelectInputArray
          isCreatable={false}
          options={userGroups}
          values={selectedGroups.sort()}
          onChange={(groups: string[]) => setSelectedGroups(groups)}
        />
      </UploadField>

      <UploadField label='Description' disabled={uploadInProgress}>
        <Form.Control
          style={{ minHeight: '200px' }}
          as="textarea"
          placeholder="Add Description"
          value={description}
          onChange={(e) => {
            setDescription(e.target.value);
            resetStatusMessages();
          }}
        />
      </UploadField>

      <UploadField label='Tags' disabled={uploadInProgress}>
        <TagSelect tags={tags} setTags={setTags} placeholderText="Add Tags" />
      </UploadField>

      <UploadField label='TLP' marker='T' disabled={uploadInProgress} >
        <TLPSelection selectedTLP={selectedTLP} onTLPChange={handleTLPChange} />
      </UploadField>

      <UploadField label='Origin' marker='T' disabled={uploadInProgress}>
        <OriginForm />
      </UploadField>

      <UploadField label='Run Pipelines' disabled={uploadInProgress}>
        <SelectPipelines
          userInfo={userInfo}
          setReactionsList={setReactionsList}
          setError={setValidationErrors}
          currentSelections={reactionsList}
        />
      </UploadField>

      <UploadField label='' rowClassName=''>
        <p>
          <sup>*</sup> This field is required.
        </p>
      </UploadField>
      <UploadField label='' rowClassName=''>
        <p>
          <sup>T</sup> This field also creates tags when specified.
        </p>
      </UploadField>

      <Row className="d-flex justify-content-center">
        <UploadField label='' rowClassName=''>
          {uploadStatus && Object.entries(uploadStatus).length > 0 && (
            <Row className="upload-bar mt-3">
              {Object.entries(uploadStatus).map(([key, value]) => (
                <OverlayTipTop key={key} tip={value.msg}>
                  {key}
                  <ProgressBarContainer name={key} value={value.progress} error={uploadError.length > 0} />
                </OverlayTipTop>
              ))}
            </Row>
          )}
          {!uploadInProgress && (
            <>
              <Row className="upload_alerts">
                <Col className="upload-field">
                  <UploadAlertBanner uploadSHA256={uploadSHA256} uploadError={uploadError} runReactionsRes={runReactionsRes} />
                </Col>
              </Row>
              <Row className="d-flex justify-content-center upload-btn">
                <Col className="upload-field">
                  <center>
                    <Button className="ok-btn" onClick={() => void handleUpload()}>
                      Upload
                    </Button>
                  </center>
                </Col>
              </Row>
            </>
          )}
        </UploadField>
      </Row>
    </>
  );
};

export default UploadForm;
