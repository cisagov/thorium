import React, { Fragment, useEffect, useState } from 'react';
import { Alert, Col, Form, Row } from 'react-bootstrap';
import { FaQuestionCircle } from 'react-icons/fa';
import { default as MarkdownHtml } from 'react-markdown';
import remarkGfm from 'remark-gfm';

// project imports
import { FieldBadge, OverlayTipRight, Subtitle } from '@components';

const ImageFieldsToolTips = {
  name: `Image name that contains only alpha-numeric characters and dashes.`,
  creator: `The user that created this image.`,
  group: `The Thorium group that can use this image.`,
  description: `A description of this image's purpose and functionality.`,
  scaler: `The scaler type that executes this image. The scaler determines where Thorium
    will execute your tool. You must have the developer role permission for this scaler.`,
  image: `The container registry path:tag for this K8s scaled image.`,
  timeout: `The max time in seconds that an image will be allowed to run before it is
    terminated.`,
  lifetime: `How long k8s scaled pods of this image will run before being scaled down. Units are
    time in seconds or number of jobs.`,
  runtime: `Average execution time for the previous 100k runs of this image. (600 default).`,
  display_type: `The format used to render image results (JSON, String, etc). Results files
    are not rendered, download links for those will be shown.`,
  spawn_limit: `The max number of tool instances that can be run simultaneously. This is useful
    when tools interact with a performance limited API or database to prevent overloading that
    resource.`,
  collect_logs: `Whether the Thorium agent collects stdout/err as logs when this image runs.`,
  generator: `Whether this image is a Thorium generator that will be respawned until it
    completes creating jobs.`,
  used_by: `The pipelines that use this image. You cannot delete an image that is used by
    a pipeline.`,
};

const ImageFieldsTemplate = {
  name: '',
  group: '',
  description: '',
  scaler: 'K8s',
  image: '',
  timeout: '',
  lifetime: '',
  display_type: '',
  spawn_limit: 'Unlimited',
  collect_logs: true,
  generator: false,
};

const ImageFieldsErrorTemplate = {
  name: 'Required',
  group: 'Required',
  image: 'Required',
  timeout: 'Required',
  display_type: 'Required',
};

// All possible Enums for the display_type field and scaler
const DisplayTypes = ['JSON', 'String', 'Table', 'Markdown', 'XML', 'HTML', 'Image', 'Disassembly', 'Hidden', 'Custom'];
const ScalerTypes = ['K8s', 'BareMetal', 'External', 'Windows', 'Kvm'];

const formatLifetimeDisplay = (lifetime) => {
  if (lifetime == null) {
    return lifetime;
  } else if (lifetime.counter && lifetime.counter == 'time') {
    return `${lifetime.amount} Seconds`;
  } else if (lifetime.counter && lifetime.counter == 'jobs') {
    return `${lifetime.amount} Jobs`;
  }
  return lifetime;
};

const DisplayImageFields = ({ image }) => {
  return (
    <Fragment>
      <Row>
        <Col className={'field-name-col'}>
          <OverlayTipRight tip={ImageFieldsToolTips.creator}>
            <b>{'Creator'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col">
          <FieldBadge field={image['creator']} color={'#305ef2'} />
        </Col>
      </Row>
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.group}>
            <b>{'Group'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className={'field-value-col'}>
          <FieldBadge field={image['group']} color={'#6a00db'} />
        </Col>
      </Row>
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.description}>
            <b>{'Description'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col">
          <MarkdownHtml remarkPlugins={[remarkGfm]}>
            {image.description && image.description != 'null' ? image.description : ''}
          </MarkdownHtml>
        </Col>
      </Row>
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.scaler}>
            <b>{'Scaler'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col">
          <FieldBadge field={image['scaler']} color={'#7e7c7c'} />
        </Col>
      </Row>
      {image.scaler == 'K8s' && (
        <Row>
          <Col className="field-name-col">
            <OverlayTipRight tip={ImageFieldsToolTips.image}>
              <b>{'Image'}</b> <FaQuestionCircle />
            </OverlayTipRight>
          </Col>
          <Col className={'field-value-col'}>
            <FieldBadge field={image['image']} color={'#7e7c7c'} />
          </Col>
        </Row>
      )}
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.timeout}>
            <b>{'Timeout'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col">
          <FieldBadge field={image['timeout']} color={'#7e7c7c'} />
        </Col>
      </Row>
      {image.scaler == 'K8s' && (
        <Row>
          <Col className="field-name-col">
            <OverlayTipRight tip={ImageFieldsToolTips.lifetime}>
              <b>{'Lifetime'}</b> <FaQuestionCircle />
            </OverlayTipRight>
          </Col>
          <Col className="field-value-col">
            <FieldBadge field={formatLifetimeDisplay(image['lifetime'])} color={'#7e7c7c'} />
          </Col>
        </Row>
      )}
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.runtime}>
            <b>{'Runtime'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className={'field-value-col'}>
          <FieldBadge field={image['runtime']} color={'#7e7c7c'} />
        </Col>
      </Row>
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.display_type}>
            <b>{'Display Type'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className={'field-value-col'}>
          <FieldBadge field={image['display_type']} color={'#7e7c7c'} />
        </Col>
      </Row>
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.spawn_limit}>
            <b>{'Spawn Limit'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className={'field-value-col'}>
          <FieldBadge field={image.spawn_limit != 'Unlimited' ? image.spawn_limit['Basic'] : 'Unlimited'} color={'#7e7c7c'} />
        </Col>
      </Row>
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.collect_logs}>
            <b>{'Logging Enabled'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className={'field-value-col'}>
          <FieldBadge field={image['collect_logs']} color={'#7e7c7c'} />
        </Col>
      </Row>
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.generator}>
            <b>{'Generator'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className={'field-value-col'}>
          <FieldBadge field={image['generator']} color={'#7e7c7c'} />
        </Col>
      </Row>
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.used_by}>
            <b>{'Used By'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className={'field-value-col'}>
          <FieldBadge field={image['used_by']} color={'#7e7c7c'} />
        </Col>
      </Row>
    </Fragment>
  );
};

const checkCreateErrors = (image, setErrors, setHasErrors) => {
  // ----------- Error checking -----------
  // Clear errors for name, group, image, timeout, display_type
  const updatedErrors = {};
  // ------------------------- image name ------------------------
  // required
  if (image.name == '') {
    updatedErrors['name'] = 'Required';
  }
  // --------------------------- group ---------------------------
  // required
  if (image.group == '') {
    updatedErrors['group'] = 'Required';
  }
  // ---------------------- image path/tag -----------------------
  // required
  if ((image.image == '' || image.image == undefined) && image.scaler == 'K8s') {
    updatedErrors['image'] = 'Required';
  }
  // ---------------------- display type -------------------------
  // required
  if (image.display_type == '') {
    updatedErrors['display_type'] = 'Required';
  }
  // -------------------- booleans & timeout ---------------------
  if (image.scaler != 'External') {
    // -------------------------- timeout --------------------------
    // logs can't be collected by External images since those aren't run by the agent
    // timeout is an integer value with units in seconds
    if (image.timeout == '') {
      updatedErrors['timeout'] = 'Required';
    } else if (isNaN(image.timeout)) {
      updatedErrors['timeout'] = 'Timeout must be an integer number (of seconds)';
    }
  }
  if (Object.keys(updatedErrors).length > 0) {
    setHasErrors(true);
  } else {
    setHasErrors(false);
  }
  setErrors(updatedErrors);
};

// We only want the relevant image fields for editing/creating new images
const filterImageFields = (image) => {
  const fields = {};
  fields['name'] = image.name;
  fields['image'] = image.image;
  fields['group'] = image.group;
  fields['description'] = image.description;
  fields['scaler'] = image.scaler;
  fields['lifetime'] = image.lifetime;
  fields['timeout'] = image.timeout;
  fields['display_type'] = image.display_type;
  fields['spawn_limit'] = image.spawn_limit;
  fields['collect_logs'] = image.collect_logs;
  fields['generator'] = image.generator;
  return fields;
};

const updateCreateRequestImageFields = (newImageFields, setRequestImageFields) => {
  const requestImageFields = structuredClone(filterImageFields(newImageFields));
  if (requestImageFields.timeout) {
    requestImageFields.timeout = Number(requestImageFields.timeout);
  }
  requestImageFields.description = String(requestImageFields.description).trim();
  if (requestImageFields.description == '') {
    delete requestImageFields.description;
  }
  // timeout
  if (requestImageFields.timeout == '') {
    delete requestImageFields.timeout;
  }
  // lifetime
  if (requestImageFields.lifetime == '') {
    delete requestImageFields.lifetime;
  }
  if (requestImageFields.scaler != 'K8s') {
    delete requestImageFields['lifetime'];
  }
  setRequestImageFields(requestImageFields);
};

const updateEditRequestImageFields = (newImageFields, setRequestImageFields) => {
  const requestImageFields = structuredClone(filterImageFields(newImageFields));
  delete requestImageFields['name'];
  delete requestImageFields['group'];

  requestImageFields.description = String(requestImageFields.description).trim();
  if (requestImageFields.description == '') {
    requestImageFields['clear_description'] = true;
    delete requestImageFields.description;
  }

  if (requestImageFields.timeout) {
    requestImageFields.timeout = Number(requestImageFields.timeout);
  }

  // clear lifetimes if not specified in update
  if (requestImageFields.lifetime == null) {
    requestImageFields['clear_lifetime'] = true;
  }

  // image string is only required for k8s scheduled images
  if (requestImageFields.scaler != 'K8s') {
    delete requestImageFields['image'];
    delete requestImageFields['lifetime'];
    requestImageFields['clear_image'] = true;
    requestImageFields['clear_lifetime'] = true;
  }
  setRequestImageFields(requestImageFields);
};

const EditImageFields = ({ initialImage, setRequestFields, setHasErrors, showErrors }) => {
  const [image, setImage] = useState(structuredClone(filterImageFields(initialImage)));
  const [errors, setErrors] = useState({});

  // update a <images>'s <field> with new <value>
  const updateImage = (field, value, type) => {
    // make a deep copy of the image fields
    const imageCopy = structuredClone(image);
    // lifetime is special, lets handle those here
    if (field == 'lifetime') {
      const newLifetime = imageCopy['lifetime'] == null ? { counter: 'jobs' } : imageCopy['lifetime'];
      // counter is the type of lifetime, options are jobs or time in seconds
      if (type == 'counter') {
        newLifetime['counter'] = value;
        // amount is the number of jobs or time in seconds (integer)
      } else if (type == 'amount') {
        if (value == '') {
          // empty string for amount clears the field
          delete newLifetime['amount'];
        } else {
          newLifetime['amount'] = Number(value);
        }
      }
      imageCopy['lifetime'] = newLifetime;
    } else {
      // set the new value for the key
      imageCopy[field] = value;
    }
    // update the image object and trigger dom refresh
    checkCreateErrors(imageCopy, setErrors, setHasErrors);
    setImage(imageCopy);
    // need to remove optional lifetimes field if amount isn't present
    if (imageCopy.lifetime && imageCopy.lifetime.amount == null) {
      delete imageCopy.lifetime;
    }
    setRequestFields(imageCopy);
  };

  // calculate height of description field
  let descriptionHeight =
    image.description && image.description != null ? Math.max(image.description.split(/\r\n|\r|\n/).length * 25, 200) : 200;

  return (
    <Fragment>
      {/* Creator cannot be edited */}
      <Row>
        <Col className="field-name-col mb-2">
          <OverlayTipRight tip={ImageFieldsToolTips.creator}>
            <b>{'Creator'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col">
          <FieldBadge field={initialImage['creator']} color={'#305ef2'} />
        </Col>
      </Row>
      {/* Group cannot be edited */}
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.group}>
            <b>{'Group'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col mb-2">
          <FieldBadge field={image['group']} color={'#6a00db'} />
        </Col>
      </Row>
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.description}>
            <b>{'Description'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col mb-2">
          <div className="image-fields">
            <Form.Group>
              <Form.Control
                as="textarea"
                style={{ minHeight: `${descriptionHeight}px` }}
                value={image.description && image.description != 'null' ? image.description : ''}
                placeholder="describe this image"
                onChange={(e) => updateImage('description', String(e.target.value))}
              />
            </Form.Group>
          </div>
        </Col>
      </Row>
      {/* Scaler */}
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.scaler}>
            <b>{'Scaler'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col mb-2">
          <div className="image-fields">
            <Form.Group>
              <Form.Select value={image.scaler} onChange={(e) => updateImage('scaler', String(e.target.value))}>
                {ScalerTypes.sort().map((scaler) => (
                  <option key={scaler} value={scaler}>
                    {scaler}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
          </div>
        </Col>
      </Row>
      {/* Image */}
      {image.scaler == 'K8s' && (
        <Row>
          <Col className="field-name-col">
            <OverlayTipRight tip={ImageFieldsToolTips.image}>
              <b>{'Image'}</b> <FaQuestionCircle />
            </OverlayTipRight>
          </Col>
          <Col className="field-value-col mb-2">
            <div className="image-fields">
              <Form.Group>
                <Form.Control
                  type="text"
                  value={image.image ? image.image : ''}
                  placeholder="docker:latest"
                  onChange={(e) => updateImage('image', String(e.target.value).trim())}
                />
              </Form.Group>
              {errors.image && showErrors && (
                <Alert variant="danger" className="d-flex justify-content-center m-2">
                  {errors.image}
                </Alert>
              )}
            </div>
          </Col>
        </Row>
      )}
      {/* Timeout */}
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.timeout}>
            <b>{'Timeout'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col mb-2">
          <div className="image-fields">
            <Form.Group>
              <Form.Control
                placeholder="seconds"
                value={image.timeout}
                onChange={(e) => {
                  const validValue = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
                  updateImage('timeout', validValue);
                }}
              />
            </Form.Group>
            {errors.timeout && showErrors && (
              <Alert variant="danger" className="d-flex justify-content-center m-2">
                {errors.timeout}
              </Alert>
            )}
          </div>
        </Col>
      </Row>
      {/* Lifetime */}
      {image.scaler == 'K8s' && (
        <Row>
          <Col className="field-name-col">
            <OverlayTipRight tip={ImageFieldsToolTips.lifetime}>
              <b>{'Lifetime'}</b> <FaQuestionCircle />
            </OverlayTipRight>
          </Col>
          <Col className="field-value-col mb-1">
            <div className="image-fields">
              <Row>
                <Col className="resource-type-col">
                  <Form.Control
                    type="text"
                    value={image.lifetime != null && image.lifetime.amount >= 0 ? image.lifetime.amount : ''}
                    placeholder={'0'}
                    onChange={(e) => {
                      const validValue = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
                      updateImage('lifetime', validValue, 'amount');
                    }}
                  />
                </Col>
                <Col className="resource-unit-col">
                  <Form.Select
                    value={image.lifetime != null ? image.lifetime.counter : ''}
                    onChange={(e) => updateImage('lifetime', String(e.target.value), 'counter')}
                  >
                    <option value="jobs">Jobs</option>
                    <option value="time">Seconds</option>
                  </Form.Select>
                </Col>
              </Row>
            </div>
          </Col>
        </Row>
      )}
      {/* Runtime cannot be edited */}
      <Row className="mb-1">
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.runtime}>
            <b>{'Runtime'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col mb-2">
          <FieldBadge field={initialImage['runtime']} color={'#7e7c7c'} />
        </Col>
      </Row>
      {/* Display Type */}
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.display_type}>
            <b>{'Display Type'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col mb-2">
          <div className="image-fields">
            <Form.Group>
              <Form.Select
                value={image.display_type ? image.display_type : ''}
                onChange={(e) => updateImage('display_type', String(e.target.value))}
              >
                {DisplayTypes.map((displayType) => (
                  <option key={displayType} value={displayType}>
                    {displayType}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
          </div>
        </Col>
      </Row>
      {/* Spawn Limit */}
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.spawn_limit}>
            <b>{'Spawn Limit'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col mb-2">
          <div className="image-fields">
            <Form.Group>
              <Form.Control
                scroll="no"
                placeholder="Unlimited"
                value={image.spawn_limit == 'Unlimited' ? '' : image.spawn_limit['Basic']}
                onChange={(e) => {
                  const validValue = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
                  updateImage('spawn_limit', validValue == '' ? 'Unlimited' : { Basic: Number(validValue) });
                }}
              />
            </Form.Group>
          </div>
        </Col>
      </Row>
      {/* Collect Logs */}
      <Row>
        <Col className="field-name-col mt-3">
          <OverlayTipRight tip={ImageFieldsToolTips.collect_logs}>
            <b>{'Logging Enabled'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col">
          <div className="image-fields">
            <Form.Group>
              <h6>
                <Form.Check
                  type="switch"
                  id="collect-logs"
                  label=""
                  checked={image.collect_logs}
                  onChange={(e) => updateImage('collect_logs', !image.collect_logs)}
                />
              </h6>
            </Form.Group>
          </div>
        </Col>
      </Row>
      {/* Generator */}
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.generator}>
            <b>{'Generator'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col mb-2">
          <div className="image-fields">
            <Form.Group>
              <h6>
                <Form.Check
                  type="switch"
                  id="is-generator"
                  label=""
                  checked={image.generator}
                  onChange={(e) => updateImage('generator', !image.generator)}
                />
              </h6>
            </Form.Group>
          </div>
        </Col>
      </Row>
      {/* Used By cannot be edited */}
      <Row>
        <Col className="field-name-col">
          <OverlayTipRight tip={ImageFieldsToolTips.used_by}>
            <b>{'Used By'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col className="field-value-col mb-2">
          <FieldBadge field={initialImage['used_by']} color={'#7e7c7c'} />
        </Col>
      </Row>
    </Fragment>
  );
};

const CreateImageFields = ({ initialImage, groups, setRequestFields, setHasErrors, showErrors }) => {
  const [image, setImage] = useState(structuredClone(filterImageFields(initialImage)));
  const [errors, setErrors] = useState(showErrors ? ImageFieldsErrorTemplate : {});

  // Showing errors for required fields only happens after a user
  // initially tries to submit the create image request. This is controlled
  // by the parent component and that component updates the showError boolean.
  useEffect(() => {
    checkCreateErrors(image, setErrors, setHasErrors);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showErrors]);

  // update a <images>'s <field> with new <value>
  const updateImage = (field, value, type) => {
    // make a deep copy of the image fields
    const imageCopy = structuredClone(image);
    // set the new value for the key
    if (field == 'lifetime') {
      const newLifetime = imageCopy['lifetime'] == null ? { counter: 'jobs' } : imageCopy['lifetime'];
      // counter is the type of lifetime, options are jobs or time in seconds
      if (type == 'counter') {
        newLifetime['counter'] = value;
        // amount is the number of jobs or time in seconds (integer)
      } else if (type == 'amount') {
        if (value === '') {
          // empty string for amount clears the field
          delete newLifetime['amount'];
        } else {
          newLifetime['amount'] = Number(value);
        }
      }
      imageCopy['lifetime'] = newLifetime;
    } else {
      // set the new value for the key
      imageCopy[field] = value;
    }

    // show errors if boolean passed in by parent is set
    checkCreateErrors(imageCopy, setErrors, setHasErrors);
    // update the image object and trigger dom refresh
    setImage(imageCopy);
    setRequestFields(imageCopy);
  };

  return (
    <Fragment>
      <Form>
        <Form.Group>
          <Form.Label>
            <Subtitle>Name</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={ImageFieldsToolTips.name}>
            <Form.Control
              type="text"
              value={image.name}
              placeholder="name"
              onChange={(e) => {
                updateImage('name', String(e.target.value));
              }}
            />
          </OverlayTipRight>
          {errors.name && showErrors && (
            <Alert variant="danger" className="d-flex justify-content-center m-2">
              {errors.name}
            </Alert>
          )}
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>Group</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={ImageFieldsToolTips.group}>
            <Form.Select
              value={image.group ? image.group : ''}
              onChange={(e) => {
                updateImage('group', String(e.target.value));
              }}
            >
              <option value="">Select a group</option>
              {groups &&
                groups.sort().map((group) => (
                  <option key={group} value={group}>
                    {group}
                  </option>
                ))}
            </Form.Select>
            {errors.group && showErrors && (
              <Alert variant="danger" className="d-flex justify-content-center m-2">
                {errors.group}
              </Alert>
            )}
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>Description</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={ImageFieldsToolTips.description}>
            <Form.Control
              as="textarea"
              value={image.description ? image.description : ''}
              placeholder="describe this image"
              onChange={(e) => updateImage('description', String(e.target.value))}
            />
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>Scaler</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={ImageFieldsToolTips.scaler}>
            <Form.Select value={image.scaler} onChange={(e) => updateImage('scaler', String(e.target.value))}>
              {ScalerTypes.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </Form.Select>
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>Image/Tag</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={ImageFieldsToolTips.image}>
            <Form.Control
              type="text"
              value={image.image}
              disabled={image.scaler != 'K8s'}
              placeholder="docker:latest"
              onChange={(e) => {
                updateImage('image', String(e.target.value).trim());
              }}
            />
          </OverlayTipRight>
          {errors.image && showErrors && (
            <Alert variant="danger" className="d-flex justify-content-center m-2">
              {errors.image}
            </Alert>
          )}
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>Timeout (seconds)</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={ImageFieldsToolTips.timeout}>
            <Form.Control
              type="text"
              value={image.timeout}
              disabled={image.scaler == 'External'}
              placeholder="seconds"
              onChange={(e) => {
                const validValue = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
                updateImage('timeout', validValue);
              }}
            />
          </OverlayTipRight>
          {errors.timeout && showErrors && (
            <Alert variant="danger" className="d-flex justify-content-center m-2">
              {errors.timeout}
            </Alert>
          )}
        </Form.Group>
        {image.scaler == 'K8s' && (
          <Form.Group>
            <Form.Label>
              <Subtitle>Lifetime</Subtitle>
            </Form.Label>
            <OverlayTipRight tip={ImageFieldsToolTips.lifetime}>
              <Row>
                <Col className="resource-type-col">
                  <Form.Control
                    type="text"
                    value={image.lifetime != null ? image.lifetime.amount : ''}
                    placeholder={image.lifetime != null ? image.lifetime.amount : ''}
                    onChange={(e) => {
                      const validValue = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : null;
                      if (validValue !== '') {
                        updateImage('lifetime', Number(validValue), 'amount');
                      }
                    }}
                  />
                </Col>
                <Col className="resource-unit-col">
                  <Form.Select
                    value={image.lifetime != null ? image.lifetime.counter : ''}
                    onChange={(e) => updateImage('lifetime', String(e.target.value), 'counter')}
                  >
                    <option value="jobs">Jobs</option>
                    <option value="time">Seconds</option>
                  </Form.Select>
                </Col>
              </Row>
            </OverlayTipRight>
          </Form.Group>
        )}
        <Form.Group>
          <Form.Label>
            <Subtitle>Display Type</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={ImageFieldsToolTips.display_type}>
            <Form.Select
              value={image.display_type}
              onChange={(e) => {
                updateImage('display_type', String(e.target.value));
              }}
            >
              <option value="">Select a display type</option>
              {DisplayTypes.map((displayType) => (
                <option key={displayType} value={displayType}>
                  {displayType}
                </option>
              ))}
            </Form.Select>
          </OverlayTipRight>
          {errors.display_type && showErrors && (
            <Alert variant="danger" className="d-flex justify-content-center m-2">
              {errors.display_type}
            </Alert>
          )}
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>Spawn Limit</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={ImageFieldsToolTips.spawn_limit}>
            <Form.Control
              scroll="no"
              placeholder="Unlimited"
              value={image.spawn_limit == 'Unlimited' ? '' : String(image.spawn_limit['Basic'])}
              onChange={(e) => {
                const validValue = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
                updateImage('spawn_limit', validValue == '' ? 'Unlimited' : { Basic: Number(validValue) });
              }}
            />
          </OverlayTipRight>
        </Form.Group>
        <Row className="mt-3">
          <Col className="collect-log-col">
            <Subtitle>Collect Logs</Subtitle>
          </Col>
          <Col>
            <Form.Group>
              <OverlayTipRight tip={ImageFieldsToolTips.collect_logs}>
                <h6>
                  <Form.Check
                    type="switch"
                    id="collect-logs"
                    label=""
                    disabled={image.scaler == 'External'}
                    checked={image.collect_logs}
                    onChange={(e) => updateImage('collect_logs', !image.collect_logs)}
                  />
                </h6>
              </OverlayTipRight>
            </Form.Group>
          </Col>
        </Row>
        <Row>
          <Col className="collect-log-col">
            <Subtitle>Generator</Subtitle>
          </Col>
          <Col>
            <Form.Group>
              <OverlayTipRight tip={ImageFieldsToolTips.generator}>
                <h6>
                  <Form.Check
                    type="switch"
                    id="is-generator"
                    label=""
                    checked={image.generator}
                    disabled={image.scaler == 'External'}
                    onChange={(e) => updateImage('generator', !image.generator)}
                  />
                </h6>
              </OverlayTipRight>
            </Form.Group>
          </Col>
        </Row>
      </Form>
    </Fragment>
  );
};

// Component to display root fields in image creation, editing or viewing existing images
const ImageFields = ({ image, setRequestImageFields, groups, setHasErrors, showErrors, mode }) => {
  const setUpdatedImageFields = (newImageFields) => {
    if (['Create', 'Copy'].includes(mode)) {
      return updateCreateRequestImageFields(newImageFields, setRequestImageFields);
    } else {
      return updateEditRequestImageFields(newImageFields, setRequestImageFields);
    }
  };

  if (mode == 'Copy') {
    const initialImage = structuredClone(image);
    initialImage['name'] = '';
    return (
      <Fragment>
        {mode == 'Copy' && (
          <CreateImageFields
            initialImage={initialImage}
            groups={groups}
            setRequestFields={setUpdatedImageFields}
            setHasErrors={setHasErrors}
            showErrors={showErrors}
          />
        )}
      </Fragment>
    );
  } else if (mode == 'Create') {
    return (
      <Fragment>
        <CreateImageFields
          initialImage={ImageFieldsTemplate}
          groups={groups}
          setRequestFields={setUpdatedImageFields}
          setHasErrors={setHasErrors}
          showErrors={showErrors}
        />
      </Fragment>
    );
  } else if (mode == 'View') {
    return (
      <Fragment>
        <DisplayImageFields image={image} />
      </Fragment>
    );
  }
  return (
    <Fragment>
      {mode == 'Edit' && (
        <EditImageFields
          initialImage={image}
          setRequestFields={setUpdatedImageFields}
          setHasErrors={setHasErrors}
          showErrors={showErrors}
        />
      )}
    </Fragment>
  );
};

export default ImageFields;
