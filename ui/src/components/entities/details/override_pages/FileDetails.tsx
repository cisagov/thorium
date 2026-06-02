import React, { Fragment, useEffect, useMemo, useState } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import { Badge, Button, Card, Col, Form, Modal, Nav, Row, Tab } from 'react-bootstrap';
import Select from 'react-select';
import { FaFileAlt, FaTrash } from 'react-icons/fa';
import { MultiValue } from 'react-select';

// project imports
const AssociationGraph = React.lazy(() => import('@components/associations/graph/AssociationGraph'));
const Results = React.lazy(() => import('@components/pages/files/Results'));
const RunPipelines = React.lazy(() => import('@components/pages/files/reactions/RunPipelines'));
import ReactionStatus from '@components/pages/files/reactions/ReactionStatus';
import Page from '@components/pages/Page';
import Subtitle from '@components/shared/titles/Subtitle';
import Time from '@components/shared/Time';
import Download from '@components/pages/files/Download';
import Comments from '@components/pages/files/Comments';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import EditableTags from '@components/tags/EditableTags';
import { LoadingSpinner } from '@components/shared/fallback/LoadingSpinner';
import { GraphDataProvider } from '@components/associations/data/GraphDataContext';
import { useAuth } from '@utilities/auth';
import { fetchGroups } from '@utilities/fetch';
import { canModifyGroup } from '@utilities/permissions';
import { scrollToSection } from '@utilities/interactions';
import { updateURLSection } from '@utilities/url';
import { deleteSubmission, getFileDetails } from '@thorpi/files';
import type { Sample, Origin } from '@models/files';
import type { Group } from '@models/groups';
import type { Output } from '@models/results';

const ValidTabs = ['results', 'associations', 'runpipelines', 'reactionstatus', 'download', 'comments'];

const FileDetails = () => {
  const { sha256 } = useParams<{ sha256: string }>();
  const [numResults, setNumResults] = useState(0);
  const [results, setResults] = useState<Record<string, Output[]>>({});
  const [details, setDetails] = useState<Partial<Sample>>({});
  const [groupDetails, setGroupDetails] = useState<Record<string, Group>>({});
  const [viewGraph, setViewGraph] = useState(false);
  const [reactionsTabSelected, setReactionsTabSelected] = useState(false);
  const [getFileError, setGetFileError] = useState('');
  const [loading, setLoading] = useState(true);
  const [deletionStatus, setDeletionStatus] = useState('');
  const [width, setWindowWidth] = useState(0);
  const location = useLocation();
  const section =
    location.hash && ValidTabs.includes(location.hash.replace('#', '').split('-')[0]) ? location.hash.replace('#', '').split('-') : [];
  const [allowResultsHashUpdate, setAllowResultsHashUpdate] = useState(false);
  const associationInitial = useMemo(() => ({ samples: [sha256!] }), [sha256]);

  // jump to correct tab/subsection on page load
  useEffect(() => {
    const triggerPageScroll = () => {
      switch (section[0]) {
        case 'results':
          setAllowResultsHashUpdate(true);
          if (section.length >= 2) {
            const tool = section.slice(1).toString().replaceAll(',', '-');
            setTimeout(() => scrollToSection(`${section[0]}-tab-${tool}`), 1500);
          }
          break;
        case 'associations':
          setViewGraph(true);
          break;
        case 'reactionstatus':
          setReactionsTabSelected(true);
          break;
        default:
          setTimeout(() => scrollToSection(`${section[0]}-tab`), 1500);
          break;
      }
    };

    if (Array.isArray(section) && section.length) {
      triggerPageScroll();
    } else {
      setTimeout(() => window.scrollTo(0, 0), 10);
      setAllowResultsHashUpdate(true);
    }
  }, []);

  // fetch file details
  useEffect(() => {
    const fetchFileDetails = async () => {
      const reqDetails = await getFileDetails(sha256!, setGetFileError);
      setLoading(true);
      if (reqDetails) {
        setDetails(reqDetails);
      }
      setLoading(false);
    };
    void fetchFileDetails();
    void fetchGroups(setGroupDetails as (groups: Record<string, Group> | Group[] | string[]) => void, () => {}, true);
  }, [sha256, deletionStatus]);

  // track window width for responsive layout
  useEffect(() => {
    const updateDimensions = () => setWindowWidth(window.innerWidth);
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // handle tab switching with side effects
  const handleTabChange = (key: string | null) => {
    if (!key) return;
    if (key.includes('results')) {
      setAllowResultsHashUpdate(true);
    } else {
      setAllowResultsHashUpdate(false);
    }

    switch (key) {
      case 'reactionstatus':
        setReactionsTabSelected(true);
        updateURLSection(key, '');
        setViewGraph(false);
        break;
      case 'results':
        updateURLSection(key, '');
        setReactionsTabSelected(false);
        setViewGraph(false);
        break;
      case 'associations':
        updateURLSection(key, '');
        setReactionsTabSelected(false);
        setViewGraph(true);
        break;
      default:
        updateURLSection(key, '');
        setReactionsTabSelected(false);
        setViewGraph(false);
        break;
    }
  };

  return (
    <Page id="file-info" className="full-min-width" title={`File · ${sha256}`}>
      {loading && <LoadingSpinner loading={true} />}
      {!loading &&
        deletionStatus &&
        (deletionStatus == 'Success' ? (
          <AlertBanner severity={Severity.Success}>Submission deleted successfully!</AlertBanner>
        ) : (
          <AlertBanner>{deletionStatus}</AlertBanner>
        ))}
      {!loading && getFileError && getFileError != '' && <AlertBanner>{getFileError}</AlertBanner>}
      <FileInfo
        details={details}
        setDetails={setDetails}
        groupDetails={groupDetails}
        screenWidth={width}
        setDeletionStatus={setDeletionStatus}
      />
      <hr />
      <Tab.Container defaultActiveKey={Array.isArray(section) && section.length ? section[0] : 'results'} onSelect={handleTabChange}>
        <Nav variant="pills">
          <Nav.Item className="details-navitem">
            <Nav.Link className="details-navlink" eventKey="results">
              Results
            </Nav.Link>
          </Nav.Item>
          <Nav.Item className="details-navitem">
            <Nav.Link className="details-navlink" eventKey="associations">
              Associations
            </Nav.Link>
          </Nav.Item>
          <Nav.Item className="details-navitem">
            <Nav.Link className="details-navlink" eventKey="runpipelines">
              Create Reactions
            </Nav.Link>
          </Nav.Item>
          <Nav.Item className="details-navitem">
            <Nav.Link className="details-navlink" eventKey="comments">
              Comments
            </Nav.Link>
          </Nav.Item>
          <Nav.Item className="details-navitem">
            <Nav.Link className="details-navlink" eventKey="reactionstatus">
              Reaction Status
            </Nav.Link>
          </Nav.Item>
          <Nav.Link className="details-navlink" eventKey="download">
            Download
          </Nav.Link>
        </Nav>
        <Nav.Item className="details-navitem"></Nav.Item>
        <GraphDataProvider initial={associationInitial}>
          <Tab.Content>
            <Tab.Pane eventKey="results" className="mt-4">
              <Results
                sha256={sha256!}
                results={results}
                setResults={setResults}
                numResults={numResults}
                allowHashUpdate={allowResultsHashUpdate}
                setNumResults={(num: number) => setNumResults(num)}
              />
            </Tab.Pane>
            <Tab.Pane eventKey="associations" className="mt-4">
              <AssociationGraph inView={viewGraph} />
            </Tab.Pane>
            <Tab.Pane eventKey="comments" className="mt-4">
              <Comments sha256={sha256!} />
            </Tab.Pane>
            <Tab.Pane eventKey="reactionstatus" className="mt-4">
              <ReactionStatus sha256={sha256!} autoRefresh={reactionsTabSelected} />
            </Tab.Pane>
            <Tab.Pane eventKey="runpipelines" className="mt-4">
              <RunPipelines sha256={sha256!} />
            </Tab.Pane>
            <Tab.Pane eventKey="download" className="mt-4">
              <Download sha256={sha256!} />
            </Tab.Pane>
          </Tab.Content>
        </GraphDataProvider>
      </Tab.Container>
    </Page>
  );
};

interface FileInfoProps {
  details: Partial<Sample>;
  setDetails: (details: Partial<Sample>) => void;
  groupDetails: Record<string, Group>;
  screenWidth: number;
  setDeletionStatus: (status: string) => void;
}

const FileInfo = ({ details, setDetails, groupDetails, screenWidth, setDeletionStatus }: FileInfoProps) => {
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
          <EditableTags
            sha256={details.sha256!}
            tags={details && 'tags' in details ? (details.tags ?? {}) : {}}
            setDetails={setDetails}
            screenWidth={screenWidth}
          />
        </Col>
      </Row>
      <Row className="my-3">
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
        <Col xs="auto">
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
                      <p>{details.submissions?.[subIndex[selectedSub]]?.description}</p>
                    </Col>
                  </Row>
                  <Row className="lg-show-row">
                    <Row>
                      <Col className="details-col" xs={2}>
                        <Subtitle>Submitted</Subtitle>
                      </Col>
                      <Col>
                        {details.submissions?.[subIndex[selectedSub]] && (
                          <p>
                            <Time verbose>{details.submissions[subIndex[selectedSub]].uploaded}</Time>
                          </p>
                        )}
                      </Col>
                    </Row>
                    <Row>
                      <Col className="details-col" xs={2}>
                        <Subtitle>Submitter</Subtitle>
                      </Col>
                      <Col>
                        <p>{details.submissions?.[subIndex[selectedSub]]?.submitter}</p>
                      </Col>
                    </Row>
                    <Row>
                      <Col className="details-col" xs={2}>
                        <Subtitle>Groups</Subtitle>
                      </Col>
                      <Col>
                        <p>
                          {details.submissions?.[subIndex[selectedSub]] &&
                            details.submissions[subIndex[selectedSub]].groups.map((group: string, idx: number) => (
                              <Badge key={idx} pill bg="" className="bg-blue py-2 px-3">
                                {group}
                              </Badge>
                            ))}
                        </p>
                      </Col>
                    </Row>
                  </Row>
                </Col>
                <Col className="lg-hide-col">
                  <Row>
                    <Col className="details-col" xs={3}>
                      <Subtitle>Submitted</Subtitle>
                    </Col>
                    <Col>
                      <p>
                        {details.submissions && details.submissions[subIndex[selectedSub]] && (
                          <Time verbose>{details.submissions[subIndex[selectedSub]].uploaded}</Time>
                        )}
                      </p>
                    </Col>
                  </Row>
                  <Row>
                    <Col className="details-col" xs={3}>
                      <Subtitle>Submitter</Subtitle>
                    </Col>
                    <Col>
                      <p>{details.submissions?.[subIndex[selectedSub]]?.submitter}</p>
                    </Col>
                  </Row>
                  <Row>
                    <Col className="details-col" xs={3}>
                      <Subtitle>Groups</Subtitle>
                    </Col>
                    <Col>
                      <p>
                        {details.submissions &&
                          details.submissions[subIndex[selectedSub]] &&
                          details.submissions[subIndex[selectedSub]].groups.map((group, idx) => (
                            <Badge key={idx} pill bg="" className="bg-blue py-2 px-3">
                              {group}
                            </Badge>
                          ))}
                      </p>
                    </Col>
                  </Row>
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

interface OriginDataProps {
  origin: Origin;
}

const OriginData = ({ origin }: OriginDataProps) => {
  const originType = Object.keys(origin)[0];
  const originData = (origin as Record<string, Record<string, unknown>>)[originType];
  return (
    <>
      <Row>
        <Col className="origin-field-name" xs={2}>
          <Subtitle>Type</Subtitle>
        </Col>
        <Col>
          <p>{originType}</p>
        </Col>
      </Row>
      {originData &&
        typeof originData === 'object' &&
        Object.keys(originData).map((key) => {
          if (key == 'carved_origin') {
            const carvedOrigin = originData[key] as string | Record<string, Record<string, string>>;
            return (
              <Fragment key={key}>
                <br />
                {carvedOrigin == 'Unknown' && (
                  <Row>
                    <Col className="origin-field-name" xs={2}>
                      <Subtitle>Carved Type</Subtitle>
                    </Col>
                    <Col>
                      <p>{carvedOrigin}</p>
                    </Col>
                  </Row>
                )}
                {carvedOrigin != 'Unknown' && typeof carvedOrigin === 'object' && (
                  <>
                    <Row>
                      <Col className="origin-field-name" xs={2}>
                        <Subtitle>Carved Type</Subtitle>
                      </Col>
                      <Col>
                        <p>{Object.keys(carvedOrigin)[0]}</p>
                      </Col>
                    </Row>
                    {Object.keys(carvedOrigin[Object.keys(carvedOrigin)[0]]).map((carvedKey) => (
                      <Row key={carvedKey}>
                        <Col className="origin-field-name" xs={2}>
                          <Subtitle>{carvedKey}</Subtitle>
                        </Col>
                        <Col>
                          <p>{carvedOrigin[Object.keys(carvedOrigin)[0]][carvedKey]}</p>
                        </Col>
                      </Row>
                    ))}
                  </>
                )}
              </Fragment>
            );
          } else {
            return (
              <Row key={key}>
                {originData[key] != null && originData[key] != '' && (
                  <Col className="origin-field-name" xs={2}>
                    <Subtitle>{key}</Subtitle>
                  </Col>
                )}
                {originData[key] != null && originData[key] != '' && key == 'parent' && (
                  <Col>
                    <a className="origin-sha256" href={`/file/${originData[key] as string}`}>
                      {originData[key] as string}
                    </a>
                    <a className="short-origin-sha256" href={`/file/${originData[key] as string}`}>
                      {(originData[key] as string).substring(0, 20) + '...'}
                    </a>
                  </Col>
                )}
                {key != 'parent' && (
                  <Col>
                    <p>{String(originData[key])}</p>
                  </Col>
                )}
              </Row>
            );
          }
        })}
    </>
  );
};

export default FileDetails;
