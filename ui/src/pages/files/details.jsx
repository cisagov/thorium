import React, { Fragment, useEffect, useState, Suspense } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import { Alert, Badge, Button, Card, Col, Form, Modal, Nav, Row, Tab } from 'react-bootstrap';
import Select from 'react-select';
import { FaFileAlt, FaTrash } from 'react-icons/fa';

// project imports
import {
  Comments,
  Download,
  EditableTags,
  OverlayTipTop,
  ReactionStatus,
  Related,
  Results,
  RunPipelines,
  Subtitle,
  Page,
  Time,
} from '@components';
import { fetchGroups, isGroupAdmin, useAuth, updateURLSection, scrollToSection } from '@utilities';
import { deleteSubmission, getFileDetails } from '@thorpi';

const ValidTabs = ['results', 'related', 'runpipelines', 'reactionstatus', 'download', 'comments'];

const FileDetailsContainer = () => {
  const { sha256 } = useParams();
  // used to display number of tool results in the top card
  const [numResults, setNumResults] = useState(0);
  const [results, setResults] = useState({});
  const [details, setDetails] = useState({});
  const [groupDetails, setGroupDetails] = useState({});
  const [reactionsTabSelected, setReactionsTabSelected] = useState(false);
  const [getFileError, setGetFileError] = useState('');
  const [listGroupsError, setListGroupsError] = useState('');
  const [deletionStatus, setDeletionStatus] = useState('');
  const [width, setWindowWidth] = useState(0);
  const location = useLocation();
  const section =
    location.hash && ValidTabs.includes(location.hash.replace('#', '').split('-')[0]) ? location.hash.replace('#', '').split('-') : [];
  const [allowResultsHashUpdate, setAllowResultsHashUpdate] = useState(false);

  // jump to correct tab/subsection when page is loaded
  useEffect(() => {
    const triggerPageScroll = () => {
      // check if result ID was provided within the location hash
      switch (section[0]) {
        case 'results':
          // length will be 2 when a tool is provided
          setAllowResultsHashUpdate(true);
          // scroll to specific result
          if (section.length >= 2) {
            const tool = section.slice(1).toString().replaceAll(',', '-');
            // hard coded 2.5 second load time. In theory this shouldn't be needed, but
            // for some reason tags aren't fully loaded when its supposed to be?
            setTimeout(() => scrollToSection(`${section[0]}-tab-${tool}`), 1500);
          }
          break;
        case 'reactionstatus':
          setReactionsTabSelected(true);
          break;
        default:
          // other tab section hash locations
          setTimeout(() => scrollToSection(`${section[0]}-tab`), 1500);
          break;
      }
    };

    // scroll when section is specified
    if (Array.isArray(section) && section.length) {
      // scroll to section
      triggerPageScroll();
    } else {
      // this fixes issues when going back to file listing, resizing screen and revisiting page
      setTimeout(() => window.scrollTo(0, 0), 10);
      // when no section specified, results tab is default and
      // we should allow setting of hash during scrolling
      setAllowResultsHashUpdate(true);
    }
  }, []);

  // initial request for file details
  useEffect(() => {
    const fetchFileDetails = async () => {
      const reqDetails = await getFileDetails(sha256, setGetFileError);
      if (reqDetails) {
        setDetails(reqDetails);
      }
    };
    fetchFileDetails();
    fetchGroups(setGroupDetails, setListGroupsError, true);

    // removing sha256 will cause details header not to update when you follow links to other files
  }, [sha256, deletionStatus]);

  // programmatically get width on screen change
  useEffect(() => {
    const updateDimensions = () => {
      const width = window.innerWidth;
      setWindowWidth(width);
    };
    // initialize to current page width
    updateDimensions();
    // add event listener to trigger new width on change
    window.addEventListener('resize', updateDimensions);

    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Update selected tab to trigger side effects within components
  const handleTabChange = (key) => {
    if (key.includes('results')) {
      // enable hash updates for results
      setAllowResultsHashUpdate(true);
    } else {
      // tabs other than results should disable hash updating
      setAllowResultsHashUpdate(false);
    }

    switch (key) {
      case 'reactionstatus':
        // Only update reaction status when tab is selected. This triggers a request
        // for reaction statuses for each group and so early hydration should be avoided
        setReactionsTabSelected(true);
        updateURLSection(key, null);
        break;
      case 'results':
        updateURLSection(key, '');
        setReactionsTabSelected(false);
        break;
      default:
        // tabs other than results should disable hash updating
        updateURLSection(key, null);
        setReactionsTabSelected(false);
        break;
    }
  };

  return (
    <Page id="file-info" className="full-min-width" title={`File · ${sha256}`}>
      {getFileError && deletionStatus == 'Success' ? (
        <Alert variant="success" className="d-flex justify-content-center">
          Submission deleted successfully!
        </Alert>
      ) : getFileError || listGroupsError ? (
        <Alert variant="warning" className="d-flex justify-content-center">
          {getFileError + listGroupsError}
        </Alert>
      ) : (
        <>
          {deletionStatus == 'Success' && (
            <Alert variant="success" className="d-flex justify-content-center">
              Submission deleted successfully!
            </Alert>
          )}
          {deletionStatus && deletionStatus != 'Success' && (
            <Alert variant="danger" className="d-flex justify-content-center">
              {deletionStatus}
            </Alert>
          )}
          <Suspense fallback={<h1>loading...</h1>}>
            <FileInfo
              details={details}
              setDetails={setDetails}
              groupDetails={groupDetails}
              screenWidth={width}
              setDeletionStatus={setDeletionStatus}
            />
          </Suspense>
          <hr />
          <Tab.Container defaultActiveKey={Array.isArray(section) && section.length ? section[0] : 'results'} onSelect={handleTabChange}>
            <Nav variant="pills">
              <Nav.Item className="details-navitem">
                <Nav.Link className="details-navlink" eventKey="results">
                  Results
                </Nav.Link>
              </Nav.Item>
              <Nav.Item className="details-navitem">
                <Nav.Link className="details-navlink" eventKey="related">
                  Related
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
            <Tab.Content>
              <Tab.Pane eventKey="results" className="mt-4">
                <Results
                  sha256={sha256}
                  results={results}
                  setResults={setResults}
                  numResults={numResults}
                  allowHashUpdate={allowResultsHashUpdate}
                  setNumResults={(num) => setNumResults(num)}
                />
              </Tab.Pane>
              <Tab.Pane eventKey="related" className="mt-4">
                <Related sha256={sha256} results={results} submissions={details.submissions} />
              </Tab.Pane>
              <Tab.Pane eventKey="comments" className="mt-4">
                <Comments sha256={sha256} />
              </Tab.Pane>
              <Tab.Pane eventKey="reactionstatus" className="mt-4">
                <ReactionStatus sha256={sha256} autoRefresh={reactionsTabSelected} />
              </Tab.Pane>
              <Tab.Pane eventKey="runpipelines" className="mt-4">
                <RunPipelines sha256={sha256} />
              </Tab.Pane>
              <Tab.Pane eventKey="download" className="mt-4">
                <Download sha256={sha256} />
              </Tab.Pane>
            </Tab.Content>
          </Tab.Container>
        </>
      )}
    </Page>
  );
};

const FileInfo = ({ details, setDetails, groupDetails, screenWidth, setDeletionStatus }) => {
  const { userInfo } = useAuth();
  // list of submissions
  const [subs, setSubs] = useState([]);
  // id of selected sub
  const [selectedSub, setSelectedSub] = useState(0);
  // key value pair of submission id and index
  const [subIndex, setSubIndex] = useState({});
  // number of submissions
  const [subSize, setSubSize] = useState(0);
  const [deleteGroups, setDeleteGroups] = useState([]);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [disableConfirmButton, setDisableConfirmButton] = useState(false);
  const [deletePermissions, setDeletePermissions] = useState({});
  const [groupPermissions, setGroupPermissions] = useState({});

  useEffect(() => {
    const sortAndSetSubmissions = (details) => {
      const unsortedSubs = {};
      const generalDeletePermissions = {};
      const groupDeletePermissions = {};

      // Get permissions based solely on group roles
      for (const group of Object.values(groupDetails)) {
        if (isGroupAdmin(group, userInfo)) {
          groupDeletePermissions[group.name] = true;
        } else {
          groupDeletePermissions[group.name] = false;
        }
      }
      setGroupPermissions(groupDeletePermissions);

      // Determine general overall permission to delete something
      // based on submitter status and group roles.
      for (const [key, value] of Object.entries(details.submissions)) {
        unsortedSubs[value.id] = parseInt(key);
        if (value.submitter == userInfo.username) {
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
      setSubs(details.submissions);
      setSubSize(details.submissions.length);
      setSubIndex(unsortedSubs);
      setSelectedSub(details.submissions[0].id);
      setDeletePermissions(generalDeletePermissions);
    };

    // if submission data exists
    if (details.submissions) {
      sortAndSetSubmissions(details);
    }
  }, [details, userInfo, groupDetails]);

  // handle removal of files/submissions using trash button
  const handleRemoveClick = async () => {
    const submission = details.submissions && details.submissions[subIndex[selectedSub]] && details.submissions[subIndex[selectedSub]].id;
    const res = await deleteSubmission(details.sha256, submission, deleteGroups, setDeletionStatus);
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
    setDeleteGroups(details.submissions[subIndex[selectedSub]].groups);
  };

  const handleCloseDeleteModal = () => {
    setShowDeleteModal(false);
  };

  const groupDeleteChanged = (event) => {
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
                    {String(details.sha256).length > 30 ? details.sha256.substring(0, 30) + '...' : details.sha256}
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
            sha256={details.sha256}
            tags={details && 'tags' in details ? details.tags : details}
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
            value={details.submission && details.submissions[selectedSub]}
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
              size="md"
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
                      return groupPermissions[group] || details.submissions[subIndex[selectedSub]].submitter == userInfo.username;
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
                      return groupPermissions[group] || details.submissions[subIndex[selectedSub]].submitter == userInfo.username;
                    })
                    .map((group) => ({ value: group, label: group }))
                }
                onChange={groupDeleteChanged}
              ></Select>
            </Modal.Body>
            <Modal.Footer className="d-flex justify-content-center">
              <Button className="danger-btn" onClick={handleRemoveClick} disabled={disableConfirmButton}>
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
                      <p>
                        {details.submissions && details.submissions[subIndex[selectedSub]] && details.submissions[subIndex[selectedSub]].id}
                      </p>
                    </Col>
                  </Row>
                  <Row className="flex-nowrap">
                    <Col className="details-col" xs={2}>
                      <Subtitle>Filename</Subtitle>
                    </Col>
                    <Col xs={9} className="flex-wrap">
                      <p>
                        {details.submissions &&
                          details.submissions[subIndex[selectedSub]] &&
                          details.submissions[subIndex[selectedSub]].name}
                      </p>
                    </Col>
                  </Row>
                  <Row>
                    <Col xs={2} className="details-col">
                      <Subtitle>Description</Subtitle>
                    </Col>
                    <Col xs={9} className="flex-wrap">
                      <p>
                        {details.submissions &&
                          details.submissions[subIndex[selectedSub]] &&
                          details.submissions[subIndex[selectedSub]].description}
                      </p>
                    </Col>
                  </Row>
                  <Row className="lg-show-row">
                    <Row>
                      <Col className="details-col" xs={2}>
                        <Subtitle>Submitted</Subtitle>
                      </Col>
                      <Col>
                        {details.submissions && details.submissions[subIndex[selectedSub]] && (
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
                        <p>
                          {details.submissions &&
                            details.submissions[subIndex[selectedSub]] &&
                            details.submissions[subIndex[selectedSub]].submitter}
                        </p>
                      </Col>
                    </Row>
                    <Row>
                      <Col className="details-col" xs={2}>
                        <Subtitle>Groups</Subtitle>
                      </Col>
                      <Col>
                        <p>
                          {details.submissions &&
                            details.submissions[subIndex[selectedSub]] &&
                            details.submissions[subIndex[selectedSub]].groups.map((group, idx) => (
                              <Badge key={idx} pill variant="" className="bg-blue py-2 px-3">
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
                      <p>
                        {details.submissions &&
                          details.submissions[subIndex[selectedSub]] &&
                          details.submissions[subIndex[selectedSub]].submitter}
                      </p>
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
                details.submissions[subIndex[selectedSub]].origin != 'None' && (
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

const OriginData = ({ origin }) => {
  // get the origin type
  const originType = Object.keys(origin)[0];
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
      {origin &&
        origin[originType] &&
        Object.keys(origin[originType]).map((key) => {
          if (key == 'carved_origin') {
            const carvedOrigin = origin[originType][key];
            return (
              <>
                <br />
                {carvedOrigin == 'Unknown' && (
                  <React.Fragment key={key}>
                    <Row>
                      <Col className="origin-field-name" xs={2}>
                        <Subtitle>Carved Type</Subtitle>
                      </Col>
                      <Col>
                        <p>{carvedOrigin}</p>
                      </Col>
                    </Row>
                  </React.Fragment>
                )}
                {carvedOrigin != 'Unknown' && (
                  <React.Fragment key={key}>
                    <Row>
                      <Col className="origin-field-name" xs={2}>
                        <Subtitle>Carved Type</Subtitle>
                      </Col>
                      <Col>
                        <p>{Object.keys(carvedOrigin)[0]}</p>
                      </Col>
                    </Row>
                    {carvedOrigin != 'Unknown' &&
                      Object.keys(carvedOrigin[Object.keys(carvedOrigin)[0]]).map((carvedKey) => (
                        <Row key={carvedKey}>
                          <Col className="origin-field-name" xs={2}>
                            <Subtitle>{carvedKey}</Subtitle>
                          </Col>
                          <Col>
                            <p>{carvedOrigin[Object.keys(carvedOrigin)[0]][carvedKey]}</p>
                          </Col>
                        </Row>
                      ))}
                  </React.Fragment>
                )}
              </>
            );
          } else {
            return (
              <Row key={key}>
                {origin[originType][key] != null && origin[originType][key] != '' && (
                  <Col className="origin-field-name" xs={2}>
                    <Subtitle>{key}</Subtitle>
                  </Col>
                )}
                {origin[originType][key] != null && origin[originType][key] != '' && key == 'parent' && (
                  <Col>
                    <a className="origin-sha256" href={`/file/${origin[originType][key]}`}>
                      {origin[originType][key]}
                    </a>
                    <a className="short-origin-sha256" href={`/file/${origin[originType][key]}`}>
                      {origin[originType][key].substring(0, 20) + '...'}
                    </a>
                  </Col>
                )}
                {key != 'parent' && (
                  <Col>
                    <p>{origin[originType][key]}</p>
                  </Col>
                )}
              </Row>
            );
          }
        })}
    </>
  );
};

export default FileDetailsContainer;
