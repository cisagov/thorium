import React, { Fragment, useEffect, useState } from 'react';
import { Col, Form, Row } from 'react-bootstrap';
import { FaQuestionCircle } from 'react-icons/fa';

// project imports
import { FieldBadge, OverlayTipRight, Subtitle } from '@components';

const SecToolTips = {
  self: `Runtime security context settings. Only admins can adjust these image settings.`,
  user: `The group ID used to run this image. This is an admin only feature.`,
  group: `Whether privilege escalation is allowed when executing the image. Default: false.`,
  allow_privilege_escalation: `Whether privilege escalation is allowed when executing the image. 
    Default: false.`,
};

const SecurityContextTemplate = {
  user: '',
  group: '',
  allow_privilege_escalation: false,
};

const updateEditRequest = (securityContext, setRequestSecurityContext) => {
  const requestSecurityContext = structuredClone(securityContext);
  if (requestSecurityContext.user == null || requestSecurityContext.user === '') {
    delete requestSecurityContext.user;
    requestSecurityContext['clear_user'] = true;
  } else {
    delete requestSecurityContext['clear_user'];
    requestSecurityContext.user = Number(requestSecurityContext.user);
  }
  if (requestSecurityContext.group == null || requestSecurityContext.group === '') {
    delete requestSecurityContext.group;
    requestSecurityContext['clear_group'] = true;
  } else {
    delete requestSecurityContext['clear_group'];
    requestSecurityContext.group = Number(requestSecurityContext.group);
  }
  setRequestSecurityContext(requestSecurityContext);
};

const updateCreateRequest = (securityContext, setRequestSecurityContext) => {
  const requestSecurityContext = structuredClone(securityContext);

  if (requestSecurityContext.user == null || requestSecurityContext.user == '') {
    delete requestSecurityContext.user;
    requestSecurityContext['clear_user'] = true;
  } else {
    requestSecurityContext.user = Number(requestSecurityContext.user);
  }
  if (requestSecurityContext.group == null || requestSecurityContext.group == '') {
    delete requestSecurityContext.group;
    requestSecurityContext['clear_group'] = true;
  } else {
    requestSecurityContext.group = Number(requestSecurityContext.group);
  }
  setRequestSecurityContext(requestSecurityContext);
};

const DisplaySecurityContext = ({ securityContext }) => {
  return (
    <Fragment>
      <Row>
        <Col style={{ flex: 0.1 }}></Col>
        <Col style={{ flex: 2.2 }}>
          <em>{`user: `}</em>
        </Col>
        <Col style={{ flex: 12.5 }}>
          <FieldBadge field={securityContext.user} color={'DarkRed'} />
        </Col>
      </Row>
      <Row>
        <Col style={{ flex: 0.1 }}></Col>
        <Col style={{ flex: 2.2 }}>
          <em>{`group: `}</em>
        </Col>
        <Col style={{ flex: 12.5 }}>
          <FieldBadge field={securityContext.group} color={'DarkRed'} />
        </Col>
      </Row>
      <Row>
        <Col style={{ flex: 0.1 }}></Col>
        <Col style={{ flex: 2.2 }}>
          <em>{`allow_privilege_escalation: `}</em>
        </Col>
        <Col style={{ flex: 12.5 }}>
          <FieldBadge field={securityContext.allow_privilege_escalation} color={'DarkRed'} />
        </Col>
      </Row>
    </Fragment>
  );
};

const EditSecurityContext = ({ securityContext, setRequestFields, disabled }) => {
  return (
    <Row>
      <Col style={{ flex: 0.2 }}></Col>
      <Col style={{ flex: 1.25 }}></Col>
      <Col style={{ flex: 8 }}>
        <SecurityContextFields initialSecurityContext={securityContext} setRequestFields={setRequestFields} disabled={disabled} />
      </Col>
    </Row>
  );
};

const CreateSecurityContext = ({ securityContext, setRequestFields, disabled }) => {
  return (
    <Fragment>
      <SecurityContextFields initialSecurityContext={securityContext} setRequestFields={setRequestFields} disabled={disabled} />
    </Fragment>
  );
};

const SecurityContextFields = ({ initialSecurityContext, setRequestFields, disabled }) => {
  const [securityContext, setSecurityContext] = useState(structuredClone(initialSecurityContext));

  // update a <dependency>'s <key> with new <value>
  const updateSecurityContext = (key, value) => {
    // make a deep copy of the dependency
    const securityContextCopy = structuredClone(securityContext);
    // set the new value for the key
    securityContextCopy[key] = value;
    // update the dependency object and trigger dom refreshsetRequestContext
    setSecurityContext(securityContextCopy);
    setRequestFields(securityContextCopy);
  };

  // this is needed for onload when cloning from an existing image
  useEffect(() => {
    setRequestFields(initialSecurityContext);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Fragment>
      <Form.Group className="image-fields">
        <Form.Label>
          <Subtitle>Run As User</Subtitle>
        </Form.Label>
        <OverlayTipRight tip={SecToolTips.user}>
          <Form.Control
            type="text"
            value={securityContext.user == null ? '' : securityContext.user}
            placeholder="99999"
            disabled={disabled}
            onChange={(e) => {
              const validValue = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
              updateSecurityContext('user', String(validValue));
            }}
          />
        </OverlayTipRight>
      </Form.Group>
      <Form.Group className="image-fields">
        <Form.Label>
          <Subtitle>Run As Group</Subtitle>
        </Form.Label>
        <OverlayTipRight tip={SecToolTips.group}>
          <Form.Control
            type="text"
            value={securityContext.group == null ? '' : securityContext.group}
            placeholder="99999"
            disabled={disabled}
            onChange={(e) => {
              const validValue = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
              updateSecurityContext('group', String(validValue));
            }}
          />
        </OverlayTipRight>
      </Form.Group>
      <Row className="image-fields">
        <Col style={{ maxWidth: '250px' }}>
          <Subtitle>Allow Privilege Escalation</Subtitle>
        </Col>
        <Col>
          <OverlayTipRight tip={SecToolTips.allow_privilege_escalation}>
            <Form.Group>
              <h6>
                <Form.Check
                  type="switch"
                  id="allow-escalation"
                  label=""
                  checked={securityContext.allow_privilege_escalation}
                  disabled={disabled}
                  onChange={(e) => updateSecurityContext('allow_privilege_escalation', !securityContext.allow_privilege_escalation)}
                />
              </h6>
            </Form.Group>
          </OverlayTipRight>
        </Col>
      </Row>
    </Fragment>
  );
};

const ImageSecurityContext = ({ securityContext, setRequestSecurityContext, mode, disabled }) => {
  // provide the edit/create components with a callback to update a
  // request formatted dependencies object
  const setUpdatedSecurityContext = (newSecurityContext) => {
    if (['Create', 'Copy'].includes(mode)) {
      return updateCreateRequest(newSecurityContext, setRequestSecurityContext);
    } else {
      return updateEditRequest(newSecurityContext, setRequestSecurityContext);
    }
  };

  if (mode == 'Copy') {
    return (
      <Row>
        <Col className="title-col">
          <h5>Security Context</h5>
        </Col>
        <Col className="field-col">
          <CreateSecurityContext securityContext={securityContext} setRequestFields={setUpdatedSecurityContext} disabled={disabled} />
        </Col>
      </Row>
    );
  } else if (mode == 'Create') {
    return (
      <Row>
        <Col className="title-col">
          <h5>Security Context</h5>
        </Col>
        <Col className="field-col">
          <CreateSecurityContext
            securityContext={SecurityContextTemplate}
            setRequestFields={setUpdatedSecurityContext}
            disabled={disabled}
          />
        </Col>
      </Row>
    );
  }

  return (
    <Fragment>
      <Row>
        <Col>
          <OverlayTipRight tip={SecToolTips.self}>
            <b>{'Security Context'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
      </Row>
      {mode == 'View' && <DisplaySecurityContext securityContext={securityContext} />}
      {mode == 'Edit' && (
        <EditSecurityContext securityContext={securityContext} setRequestFields={setUpdatedSecurityContext} disabled={disabled} />
      )}
    </Fragment>
  );
};

export default ImageSecurityContext;
