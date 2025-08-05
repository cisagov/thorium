import React, { Fragment, useEffect, useState } from 'react';
import { Accordion, Alert, Badge, Button, ButtonToolbar, ButtonGroup, Container, Col, Form, Modal, Row } from 'react-bootstrap';
import { FaQuestionCircle } from 'react-icons/fa';
import { default as MarkdownHtml } from 'react-markdown';
import remarkGfm from 'remark-gfm';

// project imports
import {
  FieldBadge,
  orderComparePipeline,
  LoadingSpinner,
  OverlayTipBottom,
  OverlayTipLeft,
  OverlayTipRight,
  SimpleSubtitle,
  Title,
  Page,
} from '@components';
import { getGroupRole, getThoriumRole, fetchGroups, useAuth } from '@utilities';
import { createPipeline, deletePipeline, listPipelines, updatePipeline } from '@thorpi';

const Pipelines = () => {
  const [loading, setLoading] = useState(false);
  const [pipelines, setPipelines] = useState([]);
  const [groups, setGroups] = useState({});
  const { userInfo, checkCookie } = useAuth();

  // get detailed pipeline info for pipelines in each group
  const fetchPipelines = async () => {
    setLoading(true);
    const allPipelines = [];
    // loop through each group to get owned pipelines
    for (const group of Object.keys(groups)) {
      const groupPipelines = await listPipelines(group, checkCookie, true, null, 1000);
      if (groupPipelines) {
        allPipelines.push(...groupPipelines);
      }
    }
    setPipelines(allPipelines);
    setLoading(false);
  };

  // need user's group roles to validate permissions to create/edit/delete pipelines
  useEffect(() => {
    fetchGroups(setGroups, null, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // need groups to get a list of pipelines
  useEffect(() => {
    fetchPipelines();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [groups]);

  /**
   * Update a Thorium pipeline
   * @param {string} name The name of the pipeline
   * @param {string} group The target pipeline's group
   * @param {object} pipelineOrder Json formatted image order
   * @param {number} pipelineSla The pipeline SLA in seconds
   * @param {string} pipelineDescription Description of the pipeline
   * @param {object} setUpdateError hook for setting request update errors
   * @returns {object} async promise for pending request
   */
  async function handlePipelineUpdate(name, group, pipelineOrder, pipelineSla, pipelineDescription, setUpdateError) {
    // build update request body
    const data = {};
    if (pipelineSla) {
      if (!isNaN(pipelineSla) && parseInt(pipelineSla) > 0) {
        data['sla'] = parseInt(pipelineSla);
      } else {
        setUpdateError('SLA must be a positive integer value');
        return;
      }
    }
    if (pipelineOrder) {
      try {
        data['order'] = JSON.parse(pipelineOrder);
      } catch (err) {
        setUpdateError('Image order must be valid JSON');
        return;
      }
    }
    if (pipelineDescription != '') {
      data['description'] = pipelineDescription;
    } else {
      data['clear_description'] = true;
    }

    if (await updatePipeline(group, name, data, setUpdateError)) {
      fetchPipelines();
    }
  }

  // Display the delete pipeline button and implement deletion
  const DeletePipelineButton = ({ pipeline }) => {
    const [showDeleteModal, setShowDeleteModal] = useState(false);
    const [deleteError, setDeleteError] = useState('');
    const handleCloseDeleteModal = () => {
      setShowDeleteModal(false);
      setDeleteError('');
    };
    const handleShowDeleteModal = () => setShowDeleteModal(true);

    return (
      <ButtonGroup className="d-flex justify-content-center">
        <OverlayTipBottom
          tip={`Delete this pipeline. Only Thorium admins, group owners/managers,
            or the pipeline's creator can delete a pipeline.`}
        >
          <Button className="warning-btn" onClick={handleShowDeleteModal}>
            Delete
          </Button>
        </OverlayTipBottom>
        <Modal show={showDeleteModal} onHide={handleCloseDeleteModal} keyboard={false}>
          <Modal.Header closeButton>
            <Modal.Title>Confirm deletion?</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            Do you really want to delete the <b>{pipeline.name}</b> pipeline?
            {deleteError != '' && (
              <Alert className="mt-4" variant="danger">
                <center>{deleteError}</center>
              </Alert>
            )}
          </Modal.Body>
          <Modal.Footer className="d-flex justify-content-center">
            <Button
              className="danger-btn"
              onClick={async () => {
                if (await deletePipeline(pipeline.group, pipeline.name, setDeleteError)) {
                  fetchPipelines();
                }
              }}
            >
              Confirm
            </Button>
          </Modal.Footer>
        </Modal>
      </ButtonGroup>
    );
  };

  const PipelineCountTipMessage =
    getThoriumRole(userInfo.role) == 'Admin'
      ? `There are a total of ${pipelines.length} Thorium pipelines.`
      : `There are a total of ${pipelines.length} Thorium pipelines owned by your groups.`;

  // Display pipeline accordion page headers
  const PipelineHeader = () => {
    return (
      <div className="accordion-list">
        <div>
          <h2>
            <OverlayTipRight tip={PipelineCountTipMessage}>
              <Badge bg="" className="count-badge">
                {pipelines.length}
              </Badge>
            </OverlayTipRight>
          </h2>
        </div>
        <Title>Pipelines</Title>
        <div>
          <h2>
            <CreatePipeline />
          </h2>
        </div>
      </div>
    );
  };

  // Get and display info about pipeline and allow editing by
  // privileged users.
  const PipelineInfo = ({ pipeline }) => {
    const [updateError, setUpdateError] = useState('');
    // set default pipeline
    const [pipelineOrder, setPipelineOrder] = useState(JSON.stringify(pipeline.order, null, ''));
    const [pipelineSla, setPipelineSla] = useState(pipeline.sla);
    const [pipelineDescription, setPipelineDescription] = useState(pipeline.description ? pipeline.description : '');
    const [inEditMode, setInEditMode] = useState(false);
    const pipelineTriggers = pipeline.triggers;

    // get the users role within the pipeline group
    const groupRole = getGroupRole(groups[pipeline.group], userInfo.username);
    const thoriumRole = getThoriumRole(userInfo.role);
    // user can modify if they created the pipeline or have a privileged role in Thorium
    const userCanModify =
      ((pipeline.creator == userInfo.username || ['Manager', 'Owner'].includes(groupRole)) && thoriumRole == 'Developer') ||
      thoriumRole == 'Admin';
    // creators or group managers/owners can delete pipelines even if they are not developers
    const userCanDelete = pipeline.creator == userInfo.username || ['Manager', 'Owner'].includes(groupRole) || thoriumRole == 'Admin';

    // calculate height of description field
    let descriptionHeight = pipelineDescription.split(/\r\n|\r|\n/).length * 32;
    if (descriptionHeight < 200) {
      descriptionHeight = 200;
    }

    return (
      <Form>
        <Row>
          <Col className="pipeline-header-col">
            <SimpleSubtitle>
              <b>Creator</b>
            </SimpleSubtitle>
          </Col>
          <Col className="pipeline-detail-col">
            <Badge bg="" className="bg-blue">
              {pipeline.creator}
            </Badge>
          </Col>
        </Row>
        <Row className="mt-2">
          <Col className="pipeline-header-col">
            <SimpleSubtitle>
              <b>Description</b>
            </SimpleSubtitle>
          </Col>
          <Col className="pipeline-detail-col">
            {inEditMode ? (
              <Form.Control
                as="textarea"
                style={{ minHeight: `${descriptionHeight}px` }}
                className="description-field"
                value={pipelineDescription}
                placeholder="describe this pipeline"
                onChange={(e) => setPipelineDescription(String(e.target.value))}
              />
            ) : (
              <MarkdownHtml remarkPlugins={[remarkGfm]}>{pipelineDescription}</MarkdownHtml>
            )}
          </Col>
        </Row>
        <Row className="mt-2">
          <Col className="pipeline-header-col">
            <OverlayTipRight
              tip={`The order of images to run. Image order must be
              formatted as a JSON array of strings and/or string arrays.`}
            >
              <SimpleSubtitle>
                <b>Order</b> <FaQuestionCircle className="fa-icon" />
              </SimpleSubtitle>
            </OverlayTipRight>
          </Col>
          <Col className="pipeline-detail-col">
            {inEditMode ? (
              <Form.Control
                as="textarea"
                value={pipelineOrder}
                placeholder="order"
                onChange={(e) => setPipelineOrder(String(e.target.value))}
              />
            ) : (
              <p>{pipelineOrder.toString()}</p>
            )}
          </Col>
        </Row>
        <Row className="mt-2">
          <Col className="pipeline-header-col">
            <OverlayTipRight tip={`The length of the SLA in seconds.`}>
              <SimpleSubtitle>
                <b>SLA</b> <FaQuestionCircle className="fa-icon" />
              </SimpleSubtitle>
            </OverlayTipRight>
          </Col>
          <Col className="pipeline-detail-col">
            {inEditMode ? (
              <Form.Control
                className="pipeline-field"
                type="text"
                value={pipelineSla}
                placeholder="SLA in seconds"
                onChange={(e) => setPipelineSla(String(e.target.value))}
              />
            ) : (
              <p>{pipelineSla}</p>
            )}
          </Col>
        </Row>
        <Row className="mt-2">
          <Col className="pipeline-header-col">
            <OverlayTipRight
              tip={`Automatic triggers that will cause this pipeline to run.
                Events can be configured to trigger when samples are initially uploaded or
                upon the creation of metadata tags.`}
            >
              <b>Event Triggers</b> <FaQuestionCircle />
            </OverlayTipRight>
          </Col>
          {Object.keys(pipelineTriggers).length == 0 && (
            <Col className="pipeline-detail-col">
              <FieldBadge field={'None'} color={'#7e7c7c'} />
            </Col>
          )}
        </Row>
        {Object.keys(pipelineTriggers).length > 0 &&
          Object.keys(pipelineTriggers).map((triggerName, idx) => (
            <div key={triggerName}>
              <Row>
                <Col className="trigger-indent" />
                <Col className="trigger-field">
                  <em>Trigger Name:</em>
                </Col>
                <Col className="trigger-value">
                  <FieldBadge field={triggerName} color={'#7e7c7c'} />
                </Col>
              </Row>
              {Object.keys(pipelineTriggers[triggerName]).length > 0 && Object.keys(pipelineTriggers[triggerName]).includes('Tag') && (
                <>
                  <Row>
                    <Col className="trigger-indent" />
                    <Col className="trigger-field">
                      <em>Trigger Type:</em>
                    </Col>
                    <Col className="trigger-value">
                      <FieldBadge field={'Tag'} color={'#7e7c7c'} />
                    </Col>
                  </Row>
                  <Row>
                    <Col className="trigger-indent" />
                    <Col className="trigger-field">
                      <em>Tag Types:</em>
                    </Col>
                    <Col className="trigger-value">
                      <FieldBadge field={pipelineTriggers[triggerName]['Tag']['tag_types']} color={'#7e7c7c'} />
                    </Col>
                  </Row>
                  <Row>
                    <Col className="trigger-indent" />
                    <Col className="trigger-field">
                      <em>Required:</em>
                    </Col>
                    <Col className="trigger-value">
                      {Object.keys(pipelineTriggers[triggerName]['Tag']['required']).length == 0 && (
                        <FieldBadge field={'None'} color={'#7e7c7c'} />
                      )}
                      {Object.keys(pipelineTriggers[triggerName]['Tag']['required'])
                        .sort()
                        .map((key) =>
                          pipelineTriggers[triggerName]['Tag']['required'][key].map((value) => (
                            <FieldBadge key={key} field={`${key}: ${value}`} color={'#7e7c7c'} />
                          )),
                        )}
                    </Col>
                  </Row>
                  <Row>
                    <Col className="trigger-indent" />
                    <Col className="trigger-field">
                      <em>Not:</em>
                    </Col>
                    <Col className="trigger-value">
                      {Object.keys(pipelineTriggers[triggerName]['Tag']['not']).length == 0 && (
                        <FieldBadge field={'None'} color={'#7e7c7c'} />
                      )}
                      {Object.keys(pipelineTriggers[triggerName]['Tag']['not'])
                        .sort()
                        .map((key) =>
                          pipelineTriggers[triggerName]['Tag']['not'][key].map((value) => (
                            <FieldBadge key={key} field={`${key}: ${value}`} color={'#7e7c7c'} />
                          )),
                        )}
                    </Col>
                  </Row>
                </>
              )}
              {pipelineTriggers[triggerName] == 'NewSample' && (
                <Row>
                  <Col className="trigger-indent" />
                  <Col className="trigger-field">
                    <em>Trigger Type:</em>
                  </Col>
                  <Col className="trigger-value">
                    <FieldBadge field={'NewSample'} color={'#7e7c7c'} />
                  </Col>
                </Row>
              )}
              {/* no hr for last element */}
              {Object.keys(pipelineTriggers).length - 1 != idx && <hr className="tagshr" />}
            </div>
          ))}
        {userCanDelete && (
          <Row className="mt-2">
            {updateError != '' && (
              <Alert variant="danger">
                <center>{updateError}</center>
              </Alert>
            )}
            <Col>
              {userCanModify && (
                <ButtonToolbar className="d-flex justify-content-center">
                  <ButtonGroup>
                    <OverlayTipBottom
                      tip={
                        inEditMode
                          ? `Edit this pipeline. Only Thorium admins or
                            developers with group permissions may edit pipelines.`
                          : `Cancel editing this pipeline.`
                      }
                    >
                      <Button className="secondary-btn me-1" onClick={() => setInEditMode(!inEditMode)}>
                        {inEditMode ? 'Cancel' : 'Edit'}
                      </Button>
                    </OverlayTipBottom>
                    {inEditMode ? (
                      <OverlayTipBottom
                        tip={`Update this pipeline. Only Thorium admins or
                          developers with group permissions may update pipelines.`}
                      >
                        <Button
                          className="ok-btn"
                          onClick={() =>
                            handlePipelineUpdate(
                              pipeline.name,
                              pipeline.group,
                              pipelineOrder,
                              pipelineSla,
                              pipelineDescription,
                              setUpdateError,
                            )
                          }
                        >
                          Update
                        </Button>
                      </OverlayTipBottom>
                    ) : (
                      <DeletePipelineButton pipeline={pipeline} />
                    )}
                  </ButtonGroup>
                </ButtonToolbar>
              )}
            </Col>
          </Row>
        )}
      </Form>
    );
  };

  // Container for create pipeline button and modal form
  const CreatePipeline = () => {
    const handleCloseCreateModal = () => setShowCreateModal(false);
    const handleShowCreateModal = () => setShowCreateModal(true);
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [newPipelineName, setNewPipelineName] = useState('');
    const [newPipelineDescription, setNewPipelineDescription] = useState('');
    const [newPipelineSla, setNewPipelineSla] = useState('');
    const [newPipelineOrder, setNewPipelineOrder] = useState('');
    const [newPipelineGroup, setNewPipelineGroup] = useState('');
    const [createError, setCreateError] = useState('');

    /**
     * Create a new Thorium pipeline
     * @returns {object} async promise for pipeline creation request
     */
    async function handlePipelineCreate() {
      const data = {};
      // pipeline name, order and group are required to create a pipeline
      if (newPipelineName && newPipelineOrder && newPipelineGroup) {
        data['name'] = newPipelineName;
        if (newPipelineDescription) {
          data['description'] = newPipelineDescription;
        }
        // add optional SLA argument to request body
        if (newPipelineSla != '') {
          if (!isNaN(newPipelineSla) && parseInt(newPipelineSla) > 0) {
            data['sla'] = parseInt(newPipelineSla);
          } else {
            setCreateError('SLA must be a positive integer value');
            return;
          }
        }
        try {
          data['order'] = JSON.parse(newPipelineOrder);
        } catch (err) {
          setCreateError('Image order must be valid JSON');
          return;
        }
        data['group'] = newPipelineGroup;
        if (await createPipeline(data, setCreateError)) {
          fetchPipelines();
        }
      } else {
        setCreateError('Pipeline name, group and order must be specified');
      }
    }

    const canCreatePipeline = ['Developer', 'Analyst', 'Admin'].includes(getThoriumRole(userInfo.role));
    const CreatePipelineTipMessage = canCreatePipeline
      ? `Create a new pipeline. You must be a
      Thorium developer, analyst, or admin to create a pipeline.`
      : `You must be a Thorium developer or
      admin to create a pipeline.`;

    return (
      <Fragment>
        <OverlayTipLeft tip={CreatePipelineTipMessage}>
          <Button className="ok-btn" onClick={handleShowCreateModal} disabled={!canCreatePipeline}>
            +
          </Button>
        </OverlayTipLeft>
        <Modal show={showCreateModal} onHide={handleCloseCreateModal} backdrop="static" keyboard={false}>
          <Modal.Header closeButton>
            <Modal.Title>Create New Pipeline</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Form.Group className="mb-4">
              <Form.Label>Name</Form.Label>
              <Form.Control
                type="text"
                value={newPipelineName}
                placeholder="name"
                onChange={(e) => setNewPipelineName(String(e.target.value))}
              />
              <Form.Text className="text-muted">Pipeline name can contain alpha-numeric characters and dashes.</Form.Text>
            </Form.Group>
            <Form.Group className="mb-4">
              <Form.Label>Description</Form.Label>
              <Form.Control
                as="textarea"
                className="description-field"
                value={newPipelineDescription}
                placeholder="describe this pipeline"
                onChange={(e) => {
                  setNewPipelineDescription(String(e.target.value));
                }}
              />
              <Form.Text className="text-muted">{`Describe this pipeline's functionality and intended use.`}</Form.Text>
            </Form.Group>
            <Form.Group className="mb-4 sla">
              <Form.Label>SLA</Form.Label>
              <Form.Control
                type="text"
                value={newPipelineSla}
                placeholder="640800"
                onChange={(e) => setNewPipelineSla(String(e.target.value))}
              />
              <Form.Text className="text-muted">Service level agreement in Seconds.</Form.Text>
            </Form.Group>
            <Form.Group className="mb-4">
              <Form.Label>Image Order</Form.Label>
              <Form.Control
                type="text"
                value={newPipelineOrder}
                placeholder='["tool1", ["parallel1", "parallel2"], "tool4"]'
                onChange={(e) => setNewPipelineOrder(String(e.target.value))}
              />
              <Form.Text className="text-muted">Format order as a JSON array of strings and/or string arrays.</Form.Text>
            </Form.Group>
            <Form.Group className="mb-4">
              <Form.Label>Group</Form.Label>
              <Form.Select onChange={(e) => setNewPipelineGroup(String(e.target.value))}>
                <option value="">Select a group</option>
                {Object.keys(groups)
                  .sort()
                  .map((group) => (
                    <option key={group} value={group}>
                      {group}
                    </option>
                  ))}
              </Form.Select>
              <Form.Text className="text-muted">Existing group that can access pipeline.</Form.Text>
            </Form.Group>
            {createError != '' && (
              <Alert variant="danger" className="mt-4">
                <center>{createError}</center>
              </Alert>
            )}
          </Modal.Body>
          <Modal.Footer className="d-flex justify-content-center">
            <Button className="ok-btn" onClick={() => handlePipelineCreate()}>
              Create
            </Button>
          </Modal.Footer>
        </Modal>
      </Fragment>
    );
  };

  return (
    <Page title="Pipelines · Thorium">
      <PipelineHeader className="accordion-list" />
      <LoadingSpinner loading={loading}></LoadingSpinner>
      <Accordion alwaysOpen>
        {pipelines
          .sort((a, b) => orderComparePipeline(a, b))
          .map((pipeline) => (
            <Accordion.Item key={`${pipeline.name}_${pipeline.group}`} eventKey={`${pipeline.name}_${pipeline.group}`}>
              <Accordion.Header>
                <Container className="accordion-list">
                  <Col className="accordion-item-name">
                    <div className="text">{pipeline.name}</div>
                  </Col>
                  <Col className="accordion-item-relation" />
                  <Col className="accordion-item-ownership">
                    <OverlayTipLeft tip={`This pipeline is owned by the ${pipeline.group} group.`}>
                      <small>
                        <i>{pipeline.group}</i>
                      </small>
                    </OverlayTipLeft>
                  </Col>
                </Container>
              </Accordion.Header>
              <Accordion.Body>
                <PipelineInfo pipeline={pipeline} />
              </Accordion.Body>
            </Accordion.Item>
          ))}
      </Accordion>
    </Page>
  );
};

export default Pipelines;
