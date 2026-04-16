import React, { Fragment } from 'react';
import { Button, Card, Col, Row } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import { FaChevronDown, FaChevronUp, FaRedo } from 'react-icons/fa';
import Subtitle from '@components/shared/titles/Subtitle';
import { useUpload } from './UploadContext';

const UploadStatusTable: React.FC = () => {
  const {
    uploadStatus,
    uploadStatusDropdown,
    toggleStatusDropdown,
    uploadReactions,
    uploadReactionRes,
    uploadInProgress,
    retryFileUpload,
    retrySubmitReaction,
  } = useUpload();

  return (
    <>
      <Row className="mt-1">
        <Card className="panel">
          <Row>
            <Col className="status-dropdown" md={1} />
            <Col className="status-file" md={1}>
              <Subtitle>Filename</Subtitle>
            </Col>
            <Col className="status-msg" md={1}>
              <Subtitle>Status</Subtitle>
            </Col>
            <Col className="status-percent" md={1}>
              <Subtitle>Progress</Subtitle>
            </Col>
            <Col className="status-sha-head">
              <Subtitle>SHA256</Subtitle>
            </Col>
          </Row>
        </Card>
      </Row>
      <Row className="mt-1">
        {Object.entries(uploadStatus).map(([key, value]) => (
          <Fragment key={key}>
            <Card className="panel">
              <Row>
                <Col className="status-dropdown" md={1}>
                  <Button size={'xsm' as 'sm'} variant="no-outline-secondary" onClick={() => toggleStatusDropdown(key)}>
                    {uploadStatusDropdown[key] ? <FaChevronUp /> : <FaChevronDown />}
                  </Button>
                </Col>
                <Col className="status-file" md={1}>
                  {key}
                </Col>
                <Col className={'status-msg' + (value.fileFail || value.reactionFail ? ' status-error' : '')} md={1}>
                  {value.msg}
                </Col>
                <Col className="status-percent" md={1}>
                  {value.fileFail ? '0' : value.progress}%
                </Col>
                {value.sha256 && (
                  <Link to={`/file/${value.sha256}`} target="_blank" className="status-sha-link link-text-alt">
                    <Col className="status-sha">{value.sha256}</Col>
                  </Link>
                )}
                {!value.sha256 && !uploadInProgress && (
                  <Col className="status-sha">
                    <Button size={'xsm' as 'sm'} variant="no-outline-secondary" className="redo-btn" onClick={() => retryFileUpload(key)}>
                      <FaRedo />
                    </Button>
                  </Col>
                )}
              </Row>
            </Card>
            {uploadStatusDropdown[key] &&
              uploadReactions[key] &&
              (uploadReactions[key].length === 0 ? (
                <Row className="upload-content">
                  <b>No Reactions Submitted</b>
                </Row>
              ) : (
                <div className="reaction-uploads-card">
                  <Card className="panel">
                    <Row className="reaction-row mt-1">
                      <Col md={2}>
                        <Subtitle>Pipeline</Subtitle>
                      </Col>
                      <Col md={2}>
                        <Subtitle>Group</Subtitle>
                      </Col>
                      <Col md={3}>
                        <Subtitle>Error</Subtitle>
                      </Col>
                      <Col md={2}>
                        <Subtitle>ID</Subtitle>
                      </Col>
                    </Row>
                  </Card>
                  {uploadReactionRes
                    .filter((result) => result.sha256 === value.sha256)
                    .map((val) => (
                      <Card className="panel" key={val.result.id}>
                        <Row className="reaction-row">
                          <Col md={2}>{val.result.pipeline}</Col>
                          <Col md={2}>{val.result.group}</Col>
                          <Col className={val.result.error ? 'status-error' : ''} md={3}>
                            {val.result.error && val.result.error.split(':')[1]}
                          </Col>
                          <Col>
                            {val.result.error && !uploadInProgress && (
                              <Button
                                size={'xsm' as 'sm'}
                                variant="no-outline-secondary"
                                className="redo-btn"
                                onClick={() => retrySubmitReaction(val)}
                              >
                                <FaRedo />
                              </Button>
                            )}
                            <Link target="_blank" className="link-text-alt" to={`/reaction/${val.result.group}/${val.result.id}`}>
                              {val.result.id}
                            </Link>
                          </Col>
                        </Row>
                      </Card>
                    ))}
                </div>
              ))}
          </Fragment>
        ))}
      </Row>
    </>
  );
};

export default UploadStatusTable;
