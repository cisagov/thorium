import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Button, ButtonToolbar, Card, Col, FormCheck, Modal, Row } from 'react-bootstrap';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { FaTrash } from 'react-icons/fa';

// project imports
import { LinkFields } from '@entities/browsing/shared';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import Subtitle from '@components/shared/titles/Subtitle';
import { OverlayTipTop, OverlayTipLeft } from '@components/shared/overlay/tips';
import { useAuth } from '@utilities/auth';
import { listReactions, deleteReaction } from '@thorpi/reactions';
import { getStatusBadge } from './reactions';
import type { Reaction, ReactionRunResult } from '@models/reactions';

// spec: ../files.spec.md

interface DeleteReactionAlertsProps {
  responses: ReactionRunResult[];
}

const DeleteReactionAlerts = ({ responses }: DeleteReactionAlertsProps) => {
  return (
    <>
      {responses.length > 0 &&
        responses.map((deleteResponse, idx) => (
          <Row key={idx}>
            {deleteResponse.error && <AlertBanner className="full-width">{deleteResponse.error}</AlertBanner>}
            {deleteResponse.error == '' && (
              <AlertBanner severity={Severity.Info} className="full-width">
                <span>
                  {`Successfully deleted reaction ${deleteResponse.id}`}
                  {` for pipeline ${deleteResponse.pipeline} from group ${deleteResponse.group}!`}
                </span>
              </AlertBanner>
            )}
          </Row>
        ))}
    </>
  );
};

// module-level flag to pause auto-refresh during deletion
let deleteInProgress = false;

interface ReactionStatusProps {
  sha256: string;
  autoRefresh: boolean;
}

const ReactionStatus = ({ sha256, autoRefresh }: ReactionStatusProps) => {
  const [loading, setLoading] = useState(false);
  const [reactionsList, setReactionsList] = useState<Reaction[]>([]);
  const [reactionsMap, setReactionsMap] = useState<Record<string, Reaction>>({});
  const [reactionsListSelections, setReactionsListSelections] = useState<Record<string, boolean>>({});
  const [reactionsAllSelected, setReactionsAllSelected] = useState(false);
  const { userInfo, checkCookie } = useAuth();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showDeleteItems] = useState(5);
  const [deleteReactionResponses, setDeleteReactionResponses] = useState<ReactionRunResult[]>([]);

  // fetch all reactions across user's groups
  const getReactionsList = async () => {
    setLoading(true);
    if (!deleteInProgress) {
      const reactions: Reaction[] = [];
      if (userInfo && userInfo.groups) {
        for (const group of userInfo.groups) {
          let moreReactions = true;
          let cursor: string | undefined = undefined;
          while (moreReactions) {
            const reactionsListRes = await listReactions(
              group,
              () => {
                void checkCookie();
              },
              '',
              sha256,
              true,
              cursor ?? null,
              10000,
            );
            if (reactionsListRes && 'details' in reactionsListRes) {
              reactions.push(...reactionsListRes.details);
              if (reactionsListRes.cursor == undefined) {
                moreReactions = false;
              } else {
                cursor = String(reactionsListRes.cursor);
              }
            } else {
              moreReactions = false;
            }
          }
        }
        setReactionsList(reactions);
        const newMap: Record<string, Reaction> = {};
        reactions.forEach((reaction) => (newMap[reaction.id] = reaction));
        setReactionsMap(newMap);
      }
      setLoading(false);
    }
  };

  // trigger reaction list fetch when component is viewed
  useEffect(() => {
    if (autoRefresh) {
      void getReactionsList();
      const intervalId = setInterval(() => {
        void getReactionsList();
      }, 30000);
      return () => {
        clearInterval(intervalId);
      };
    }
  }, [userInfo, sha256, autoRefresh]);

  const handleSelectionChange = (key: string) => {
    setDeleteReactionResponses([]);
    setReactionsListSelections((prevState) => {
      const newState = { ...prevState };
      newState[key] = prevState[key] == undefined ? true : !prevState[key];
      return newState;
    });
    setReactionsAllSelected(false);
  };

  const handleSelectAll = () => {
    setDeleteReactionResponses([]);
    const newSelections: Record<string, boolean> = {};
    for (const reaction of reactionsList) {
      if (reaction.id) {
        newSelections[reaction.id] = !reactionsAllSelected;
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

  // check if any selections are active
  const anySelected = (): boolean => {
    return Object.values(reactionsListSelections).some((v) => v);
  };

  // build truncated list for the delete confirmation modal
  const truncateSelections = (selections: Record<string, boolean>): string[] => {
    let truncatedList: string[] = [];
    Object.keys(selections).map((reactionID) => {
      if (selections[reactionID] && reactionsMap[reactionID]) {
        truncatedList.push(`${reactionsMap[reactionID].pipeline} : ${reactionsMap[reactionID].group}`);
      }
    });
    if (truncatedList.length > 5) {
      const truncateMsg = `${truncatedList.length - 5} more selections ...`;
      truncatedList = truncatedList.slice(0, showDeleteItems);
      truncatedList.push(truncateMsg);
    }
    return truncatedList;
  };

  // delete all selected reactions
  const handleDeleteClick = async () => {
    setShowDeleteModal(false);
    setLoading(true);
    deleteInProgress = true;
    setReactionsListSelections({});
    setReactionsAllSelected(false);

    const deleteErrors: ReactionRunResult[] = [];
    for (const reactionID of Object.keys(reactionsListSelections)) {
      if (reactionsListSelections[reactionID] && reactionsMap[reactionID]) {
        const reaction = reactionsMap[reactionID];
        const handleError = (error: string) => {
          deleteErrors.push({
            error,
            id: reaction.id,
            group: reaction.group,
            pipeline: reaction.pipeline,
          });
        };
        const res = await deleteReaction(reaction.group, reaction.id, handleError);
        if (res) {
          deleteErrors.push({
            error: '',
            id: reaction.id,
            group: reaction.group,
            pipeline: reaction.pipeline,
          });
        }
      }
    }

    setLoading(false);
    deleteInProgress = false;
    await getReactionsList();
    setDeleteReactionResponses(deleteErrors);
  };

  return (
    <div id="reactionstatus-tab" className="mx-4">
      {!loading && reactionsList.length == 0 ? (
        <>
          <AlertBanner severity={Severity.Info}>
            <h3>No Reactions Found</h3>
          </AlertBanner>
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
                        </Modal.Body>
                        <Modal.Footer className="d-flex justify-content-center">
                          <Button
                            className="danger-btn"
                            onClick={() => {
                              void handleDeleteClick();
                            }}
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
                </Card>
              </Row>
              <Row className="mt-1">
                {reactionsList.map((reaction, idx) => (
                  <Card key={`${reaction.id}_${idx}`} className="no-border">
                    <LinkFields className="no-decoration">
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-pipeline">
                        <Col>{reaction.pipeline}</Col>
                      </Link>
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-creator">
                        <Col>{reaction.creator}</Col>
                      </Link>
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-group">
                        <Col>{reaction.group}</Col>
                      </Link>
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-status">
                        <Col>{getStatusBadge(reaction.status)}</Col>
                      </Link>
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-id">
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
