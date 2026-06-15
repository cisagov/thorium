import React, { Fragment, ReactNode, useEffect, useState } from 'react';
import { Badge, Button, ButtonToolbar, Card, Col, Form, Modal, Row } from 'react-bootstrap';
import Select from 'react-select';
import { FaFileAlt, FaSitemap, FaTrash } from 'react-icons/fa';
import { MultiValue } from 'react-select';

// project imports
import Subtitle from '@components/shared/titles/Subtitle';
import Markdown from '@components/shared/syntax/Markdown';
import Time from '@components/shared/Time';
import UserAvatar from '@components/shared/UserAvatar';
import { OverlayTipBottom, OverlayTipTop } from '@components/shared/overlay/tips';
import EditableTags from '@components/tags/EditableTags';
import { useAuth } from '@utilities/auth';
import { canModifyGroup } from '@utilities/permissions';
import { deleteSubmission } from '@thorpi/files';
import type { Sample } from '@models/files';
import type { Group } from '@models/groups';
import styled from 'styled-components';
import { BuildDashboardButton, BuildDashboardResource, IconButton } from '@components/shared/buttons';
import { LuPackageSearch } from 'react-icons/lu';
import OriginData from './OriginData';

// submitter name followed by the submitter's avatar, vertically centered; the zero margin keeps the
// name aligned with the avatar despite the default paragraph bottom margin
const SubmitterLine = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;

  p {
    margin: 0;
  }
`;

const FileInfoRow = ({ title, children }: { children: ReactNode; title: string }) => {
  return (
    <Row>
      <Col className="details-col" xs={3}>
        <Subtitle>{title}</Subtitle>
      </Col>
      <Col>{children}</Col>
    </Row>
  );
};

interface FileActionsToolbarProps {
  // the file's sha256, used to seed the dashboard builder
  sha256: string;
  // opens the given file-detail tab (and scrolls the tab strip into view)
  onNavigateTab: (key: string) => void;
}

// centered toolbar rendered under the tags section: quick jumps to the Entities/Associations tabs
// plus the shared build-dashboard entry point
const FileActionsToolbar = ({ sha256, onNavigateTab }: FileActionsToolbarProps) => (
  <ButtonToolbar className="mt-3">
    <OverlayTipBottom tip="Entities Search">
      <IconButton onClick={() => onNavigateTab('entities')} aria-label="Entities Search">
        <LuPackageSearch size={20} />
      </IconButton>
    </OverlayTipBottom>
    <OverlayTipBottom tip="Associations Graph">
      <IconButton onClick={() => onNavigateTab('associations')} aria-label="Associations Graph">
        <FaSitemap size={20} />
      </IconButton>
    </OverlayTipBottom>
    <BuildDashboardButton resource={BuildDashboardResource.Sample} id={sha256} label="file" />
  </ButtonToolbar>
);

interface FileInfoProps {
  details: Partial<Sample>;
  setDetails: (details: Partial<Sample>) => void;
  groupDetails: Record<string, Group>;
  setDeletionStatus: (status: string) => void;
  // opens a file-detail tab from the toolbar rendered under the tags section
  onNavigateTab: (key: string) => void;
}

const FileInfo = ({ details, setDetails, groupDetails, setDeletionStatus, onNavigateTab }: FileInfoProps) => {
  const { userInfo } = useAuth();
  const [subs, setSubs] = useState<Sample['submissions']>([]);
  const [selectedSub, setSelectedSub] = useState<string>('');
  const [subIndex, setSubIndex] = useState<Record<string, number>>({});
  const [subSize, setSubSize] = useState(0);
  const [deleteGroups, setDeleteGroups] = useState<string[]>([]);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [disableConfirmButton, setDisableConfirmButton] = useState(false);
  const [deletePermissions, setDeletePermissions] = useState<Record<string, boolean>>({});
  const [groupPermissions, setGroupPermissions] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const sortAndSetSubmissions = (details: Partial<Sample>) => {
      const unsortedSubs: Record<string, number> = {};
      const generalDeletePermissions: Record<string, boolean> = {};
      const groupDeletePermissions: Record<string, boolean> = {};

      for (const group of Object.values(groupDetails)) {
        groupDeletePermissions[group.name] = canModifyGroup(group, userInfo!);
      }
      setGroupPermissions(groupDeletePermissions);

      for (const [key, value] of Object.entries(details.submissions!)) {
        unsortedSubs[value.id] = parseInt(key);
        if (value.submitter == userInfo!.username) {
          generalDeletePermissions[value.id] = true;
        } else {
          generalDeletePermissions[value.id] = false;
        }
        for (const group of Object.values(value.groups)) {
          if (groupDeletePermissions[group]) {
            generalDeletePermissions[value.id] = true;
          }
        }
      }
      setSubs(details.submissions!);
      setSubSize(details.submissions!.length);
      setSubIndex(unsortedSubs);
      setSelectedSub(details.submissions![0].id);
      setDeletePermissions(generalDeletePermissions);
    };

    if (details.submissions) {
      sortAndSetSubmissions(details);
    }
  }, [details, userInfo, groupDetails]);

  // handle deletion of a submission
  const handleRemoveClick = async () => {
    const submission = details.submissions && details.submissions[subIndex[selectedSub]] && details.submissions[subIndex[selectedSub]].id;
    const res = await deleteSubmission(details.sha256!, submission!, deleteGroups, setDeletionStatus);
    if (res) {
      setDeletionStatus('Success');
    }
    setDisableConfirmButton(true);
    setShowDeleteModal(false);
  };

  const handleShowDeleteModal = () => {
    setDeletionStatus('');
    setDisableConfirmButton(false);
    setShowDeleteModal(true);
    setDeleteGroups(details.submissions![subIndex[selectedSub]].groups);
  };

  const handleCloseDeleteModal = () => {
    setShowDeleteModal(false);
  };

  const groupDeleteChanged = (event: Array<{ value: string; label: string }>) => {
    if (!event.length) {
      setDisableConfirmButton(true);
    } else {
      setDisableConfirmButton(false);
    }
    setDeleteGroups(event.map((e) => e.value));
  };

  return (
    <Fragment>
      <Row>
        <Col>
          <Card className="panel">
            <Card.Body>
              <Row className="d-flex justify-content-center">
                <Col xs={1} className="info-icon me-6">
                  <img src="/ferris-scientist.png" alt="FerrisScientist" width="150" />
                </Col>
                <Col className="details-sha256 ms-4">
                  <Row className="sha-md5-alignment mt-3 hide-sha256 hide-sha256">{details.sha256}</Row>
                  <Row className="sha-md5-alignment short-sha256">
                    {String(details.sha256).length > 30 ? details.sha256!.substring(0, 30) + '...' : details.sha256}
                  </Row>
                  <Row className="sha-md5-alignment">
                    <Subtitle>SHA-256</Subtitle>
                  </Row>
                </Col>
                <Col className="details-sha-md5">
                  <Row className="sha-md5-alignment mt-3 mb-3">{details.sha1}</Row>
                  <Row className="sha-md5-alignment">
                    <Subtitle>SHA-1</Subtitle>
                  </Row>
                  <Row className="sha-md5-alignment mt-3 mb-3">{details.md5}</Row>
                  <Row className="sha-md5-alignment">
                    <Subtitle>MD5</Subtitle>
                  </Row>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col className="tags">
          <EditableTags sha256={details.sha256!} tags={details && 'tags' in details ? (details.tags ?? {}) : {}} setDetails={setDetails} />
        </Col>
      </Row>
      <FileActionsToolbar sha256={details.sha256!} onNavigateTab={onNavigateTab} />
      <Row className="mt-2 mb-4">
        <Col xs="auto" className="mt-3">
          <p>Select submission:</p>
        </Col>
        <Col className="mt-1">
          <Form.Control
            className="form-select"
            as="select"
            name="submission"
            value={details.submissions && details.submissions[subIndex[selectedSub]]?.id}
            onChange={(e) => {
              setDisableConfirmButton(true);
              setSelectedSub(e.target.value);
            }}
          >
            {subs && subs.map((sub, idx) => <option key={idx}>{sub.id}</option>)}
          </Form.Control>
        </Col>
        <Col xs="auto" className="d-flex align-items-center justify-content-center">
          <OverlayTipTop
            tip={`Delete this submission. Only system admins,
                group owners/managers, and the submitter can delete a submission.`}
          >
            <Button
              size="sm"
              variant=""
              className="icon-btn"
              disabled={!deletePermissions[selectedSub]}
              onClick={() => handleShowDeleteModal()}
            >
              <FaTrash />
            </Button>
          </OverlayTipTop>
          <Modal show={showDeleteModal} onHide={handleCloseDeleteModal} backdrop="static" keyboard={false}>
            <Modal.Header closeButton>
              <Modal.Title>Confirm deletion?</Modal.Title>
            </Modal.Header>
            <Modal.Body>
              <p>Do you really want to delete the submission:</p>
              <center>
                <p>
                  <b>{selectedSub}</b>
                </p>
              </center>
              from the following groups:
              <Select
                defaultValue={
                  details.submissions &&
                  details.submissions[subIndex[selectedSub]] &&
                  details.submissions[subIndex[selectedSub]].groups
                    .filter((group) => {
                      return groupPermissions[group] || details.submissions![subIndex[selectedSub]].submitter == userInfo!.username;
                    })
                    .map((group) => ({ value: group, label: group }))
                }
                className="basic-multi-select"
                classNamePrefix="select"
                isMulti
                options={
                  details.submissions &&
                  details.submissions[subIndex[selectedSub]] &&
                  details.submissions[subIndex[selectedSub]].groups
                    .filter((group) => {
                      return groupPermissions[group] || details.submissions![subIndex[selectedSub]].submitter == userInfo!.username;
                    })
                    .map((group) => ({ value: group, label: group }))
                }
                onChange={(newValue: MultiValue<{ value: string; label: string }>) => groupDeleteChanged([...newValue])}
              ></Select>
            </Modal.Body>
            <Modal.Footer className="d-flex justify-content-center">
              <Button
                className="danger-btn"
                onClick={() => {
                  void handleRemoveClick();
                }}
                disabled={disableConfirmButton}
              >
                Confirm
              </Button>
              <Button className="primary-btn" onClick={handleCloseDeleteModal}>
                Cancel
              </Button>
            </Modal.Footer>
          </Modal>
        </Col>
      </Row>
      <Row>
        <Col>
          <Card className="panel">
            <Card.Body>
              <Row>
                <Col xs={1} className="me-2 info-icon">
                  <FaFileAlt size="72" className="icon" />
                </Col>
                <Col className="lg-center-col" xs={6}>
                  <Row className="flex-nowrap">
                    <Col xs={2} className="details-col">
                      <Subtitle>Submission</Subtitle>
                    </Col>
                    <Col xs={9} className="flex-wrap">
                      <p>{details.submissions?.[subIndex[selectedSub]]?.id}</p>
                    </Col>
                  </Row>
                  <Row className="flex-nowrap">
                    <Col className="details-col" xs={2}>
                      <Subtitle>Filename</Subtitle>
                    </Col>
                    <Col xs={9} className="flex-wrap">
                      <p>{details.submissions?.[subIndex[selectedSub]]?.name}</p>
                    </Col>
                  </Row>
                  <Row>
                    <Col xs={2} className="details-col">
                      <Subtitle>Description</Subtitle>
                    </Col>
                    <Col xs={9} className="flex-wrap">
                      <Markdown>
                        {(() => {
                          const description = details.submissions?.[subIndex[selectedSub]]?.description;
                          return description && description !== 'null' ? description : '';
                        })()}
                      </Markdown>
                    </Col>
                  </Row>
                </Col>
                <Col>
                  <FileInfoRow title="Submitted">
                    <p>
                      {details.submissions && details.submissions[subIndex[selectedSub]] && (
                        <Time verbose>{details.submissions[subIndex[selectedSub]].uploaded}</Time>
                      )}
                    </p>
                  </FileInfoRow>
                  <FileInfoRow title="Submitter">
                    <SubmitterLine>
                      <UserAvatar username={details.submissions?.[subIndex[selectedSub]]?.submitter} size={28} />
                      <p>{details.submissions?.[subIndex[selectedSub]]?.submitter}</p>
                    </SubmitterLine>
                  </FileInfoRow>
                  <FileInfoRow title="Groups">
                    <p>
                      {details.submissions &&
                        details.submissions[subIndex[selectedSub]] &&
                        details.submissions[subIndex[selectedSub]].groups.map((group, idx) => (
                          <Badge key={idx} pill bg="" className="bg-blue py-2 px-3">
                            {group}
                          </Badge>
                        ))}
                    </p>
                  </FileInfoRow>
                </Col>
                <Col xs={2} className="details-circle">
                  <Subtitle>
                    <center>Submissions</center>
                  </Subtitle>
                  <div className="circle">{subSize}</div>
                </Col>
              </Row>
              {details.submissions &&
                details.submissions[subIndex[selectedSub]] &&
                (details.submissions[subIndex[selectedSub]].origin as unknown) != 'None' && (
                  <>
                    <Row>
                      <Col className="d-flex justify-content-center">
                        <h5>Origin</h5>
                      </Col>
                    </Row>
                    <Row>
                      <Col xs={1} className="mr-2 info-icon"></Col>
                      <Col xs={9}>
                        <OriginData origin={details.submissions[subIndex[selectedSub]].origin} />
                      </Col>
                    </Row>
                  </>
                )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Fragment>
  );
};

export default FileInfo;
