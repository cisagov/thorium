import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Alert, Button, ButtonToolbar, Card, Col, FormCheck, Modal, Row } from 'react-bootstrap';
import { FaTrash } from 'react-icons/fa';

// project imports
import { LinkFields } from '@entities/browsing/shared';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import Subtitle from '@components/shared/titles/Subtitle';
import { OverlayTipTop, OverlayTipLeft } from '@components/shared/overlay/tips';
import { useAuth } from '@utilities/auth';
import { listReactions } from '@thorpi/reactions';

// Alert component for error and info responses for component deletion
const DeleteReactionAlerts = ({ responses }) => {
  return (
    <>
      {responses.length > 0 &&
        responses.map((deleteResponse, idx) => (
          <Row key={idx}>
            {deleteResponse.error && (
              <Alert className="full-width" variant="danger">
                <center>{deleteResponse.error}</center>
              </Alert>
            )}
            {deleteResponse.error == '' && (
              <Alert className="full-width" variant="info">
                <center>
                  <span>
                    {`Successfully deleted reaction ${deleteResponse.id}`}
                    {` for pipeline ${deleteResponse.pipeline} from group ${deleteResponse.group}!`}
                  </span>
                </center>
              </Alert>
            )}
          </Row>
        ))}
    </>
  );
};

// not sure of a way to better avoid this (file) global
// Without it there isn't a good way to pause the auto refresh
// functionality.
let deleteInProgress = false;

const ReactionStatus = ({ sha256, autoRefresh }) => {
  const [loading, setLoading] = useState(false);
  const [reactionsList, setReactionsList] = useState([]);
  const [reactionsMap, setReactionsMap] = useState({});
  const [reactionsListSelections, setReactionsListSelections] = useState({});
  const [reactionsAllSelected, setReactionsAllSelected] = useState(false);
  const { userInfo, checkCookie } = useAuth();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showDeleteItems] = useState(5);
  const [deleteReactionResponses, setDeleteReactionResponses] = useState([]);

  const getReactionsList = async () => {
    setLoading(true);
    if (!deleteInProgress) {
      const reactions = [];
      if (userInfo && userInfo.groups) {
        for (const group of userInfo.groups) {
          let moreReactions = true;
          let cursor = null;
          // need to get all reactions in chunks of 100 until there are no more left
          while (moreReactions) {
            const reactionsList = await listReactions(group, checkCookie, '', sha256, true, cursor, 10000);
            if (reactionsList) {
              // add returned reactions to local reactions array
              reactions.push(...reactionsList.details);
              // if cursor is undefined there are no more reactions for this group/tag
              if (reactionsList['cursor'] == undefined) {
                moreReactions = false;
              } else {
                cursor = reactionsRes.data.cursor;
              }
            }
          }
        }
        setReactionsList(reactions);
        reactions.forEach((reaction) => (reactionsMap[reaction.id] = reaction));
        setReactionsMap(reactionsMap);
      }
      setLoading(false);
    }
  };

  // get a list of reactions by the sha256 tag
  useEffect(() => {
    // only trigger reaction status API requests when component is being viewed
    if (autoRefresh) {
      // get a reaction list for the first render
      getReactionsList();
      // now update the list every X seconds where x is the interval passed in below
      const intervalId = setInterval(() => {
        getReactionsList();
      }, 30000);
      return () => {
        clearInterval(intervalId);
      };
    }
  }, [userInfo, sha256, autoRefresh]);

  const handleSelectionChange = (key) => {
    setDeleteReactionResponses([]);
    setReactionsListSelections((prevState) => {
      const newState = { ...prevState };
      if (prevState[key] == undefined) {
        newState[key] = true;
      } else {
        newState[key] = !prevState[key];
      }
      return newState;
    });
    setReactionsAllSelected(false);
  };

  const handleSelectAll = () => {
    setDeleteReactionResponses([]);
    const newSelections = {};
    for (const reaction in reactionsList) {
      if (reactionsList[reaction].id) {
        newSelections[reactionsList[reaction].id] = !reactionsAllSelected;
      }
    }
    setReactionsListSelections(newSelections);
    setReactionsAllSelected(!reactionsAllSelected);
  };

  const handleShowDeleteModal = () => {
    setShowDeleteModal(true);
  };

  const handleCloseDeleteModal = () => {
    setShowDeleteModal(false);
  };

  const anySelected = () => {
    // Check is any of the selection items are actually selected.
    const values = Object.values(reactionsListSelections);
    for (let i = 0; i < values.length; i++) {
      if (values[i]) {
        return true;
      }
    }
    return false;
  };

  const truncateSelections = (selections) => {
    // Function for displaying truncated list of reactions that will be deleted
    let truncatedList = [];
    Object.keys(selections).map((reactionID, idx) => {
      if (selections[reactionID]) {
        truncatedList.push(`${reactionsMap[reactionID].pipeline} : 
          ${reactionsMap[reactionID].group}`);
      }
    });
    if (truncatedList.length > 5) {
      const truncateMsg = `${truncatedList.length - 5} more selections ...`;
      truncatedList = truncatedList.slice(0, showDeleteItems);
      truncatedList.push(truncateMsg);
    }
    return truncatedList;
  };

  const handleDeleteClick = async () => {
    setShowDeleteModal(false);
    setLoading(true);
    deleteInProgress = true;
    setReactionsListSelections({});
    setReactionsAllSelected(false);

    const deleteReactionList = [];
    Object.keys(reactionsListSelections).map(async (reactionID, idx) => {
      if (reactionsListSelections[reactionID]) {
        deleteReactionList.push(reactionsMap[reactionID]);
      }
    });
    const deleteErrors = await deleteReactions(deleteReactionList);
    setLoading(false);
    deleteInProgress = false;
    await getReactionsList();
    setDeleteReactionResponses(deleteErrors);
  };

  return (
    <div id="reactionstatus-tab" className="mx-4">
      {!loading && reactionsList.length == 0 ? (
        <>
          <Alert variant="" className="info">
            <Alert.Heading>
              <center>
                <h3>No Reactions Found</h3>
              </center>
            </Alert.Heading>
            <center>
              <p>Create a reaction and then check the status here</p>
            </center>
          </Alert>
        </>
      ) : (
        <>
          <LoadingSpinner loading={loading}></LoadingSpinner>
          {loading ? (
            <></>
          ) : (
            <>
              <Row>
                <Card className="panel">
                  <Row>
                    <Col className="reactions-pipeline mt-3" md={2}>
                      <Subtitle>Pipeline</Subtitle>
                    </Col>
                    <Col className="reactions-creator mt-3" md={1}>
                      <Subtitle className="mt-2">Creator</Subtitle>
                    </Col>
                    <Col className="reactions-group mt-3" md={1}>
                      <Subtitle>Group</Subtitle>
                    </Col>
                    <Col className="reactions-status mt-3" md={1}>
                      <Subtitle>Status</Subtitle>
                    </Col>
                    <Col className="reactions-id mt-3" md={3}>
                      <Subtitle>Reaction ID</Subtitle>
                    </Col>
                    <Col className="reactions-selection d-flex justify-content-end" md={1}>
                      <ButtonToolbar className="d-flex justify-content-end">
                        <OverlayTipTop
                          tip={`Delete selected reactions. Only system admins,
                          group owners/managers, and the submitter can delete a reaction.`}
                        >
                          <Button
                            size="sm"
                            className="icon-btn me-2 my-1"
                            variant=""
                            disabled={!anySelected()}
                            onClick={handleShowDeleteModal}
                          >
                            <FaTrash />
                          </Button>
                        </OverlayTipTop>
                        <OverlayTipLeft tip={'Select All Reactions'}>
                          <FormCheck onChange={handleSelectAll} className="mt-2" checked={reactionsAllSelected}></FormCheck>
                        </OverlayTipLeft>
                      </ButtonToolbar>
                      <Modal show={showDeleteModal} onHide={handleCloseDeleteModal} backdrop="static" keyboard={false}>
                        <Modal.Header closeButton>
                          <Modal.Title>Confirm deletion?</Modal.Title>
                        </Modal.Header>
                        <Modal.Body>
                          <p>Do you really want to delete the following reactions:</p>
                          {truncateSelections(reactionsListSelections).map((reactionString, idx) => {
                            return (
                              <div key={idx}>
                                <center>
                                  <b>{reactionString}</b>
                                </center>
                              </div>
                            );
                          })}
                          <center>
                            <p>
                              <b>{}</b>
                              <b>{}</b>
                            </p>
                          </center>
                        </Modal.Body>
                        <Modal.Footer className="d-flex justify-content-center">
                          <Button className="danger-btn" onClick={handleDeleteClick}>
                            Confirm
                          </Button>
                          <Button className="primary-btn" onClick={handleCloseDeleteModal}>
                            Cancel
                          </Button>
                        </Modal.Footer>
                      </Modal>
                    </Col>
                  </Row>
                </Card>
              </Row>
              <Row className="mt-1">
                {reactionsList.map((reaction, idx) => (
                  <Card key={`${reaction.id}_${idx}`} className="no-border">
                    <LinkFields className="no-decoration">
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-pipeline" md={2}>
                        <Col>{reaction.pipeline}</Col>
                      </Link>
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-creator" md={1}>
                        <Col>{reaction.creator}</Col>
                      </Link>
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-group" md={1}>
                        <Col>{reaction.group}</Col>
                      </Link>
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-status" md={1}>
                        <Col>{getStatusBadge(reaction.status)}</Col>
                      </Link>
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-id" md={3}>
                        <Col>{reaction.id}</Col>
                      </Link>
                      <Col className="reactions-selection" md={1}>
                        <FormCheck
                          onChange={() => {
                            handleSelectionChange(reaction.id);
                          }}
                          checked={reactionsListSelections[reaction.id] ? reactionsListSelections[reaction.id] : false}
                        ></FormCheck>
                      </Col>
                    </LinkFields>
                  </Card>
                ))}
              </Row>
            </>
          )}
        </>
      )}
      <Row className="mt-2">
        <DeleteReactionAlerts responses={deleteReactionResponses} />
      </Row>
      <br />
      <br />
    </div>
  );
};

export default ReactionStatus;
