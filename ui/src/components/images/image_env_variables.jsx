import React, { Fragment, useEffect, useState } from 'react';
import { Col, Row } from 'react-bootstrap';
import { FaQuestionCircle } from 'react-icons/fa';

// project imports
import { FieldBadge, OverlayTipRight, SelectableDictionary } from '@components';

const EnvironmentToolTip = `Environment variables that get mapped into the running image.`;

const DisplayEnvironmentVars = ({ environmentVars }) => {
  return (
    <Fragment>
      <Row>
        <Col style={{ flex: 2.5 }}>
          <OverlayTipRight tip={EnvironmentToolTip}>
            <b>{'Environment'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col style={{ flex: 12.5 }}>
          {!Object.entries(environmentVars).length && <FieldBadge field={environmentVars} color={'#7e7c7c'} />}
        </Col>
      </Row>
      <Row>
        {Object.entries(environmentVars).length > 0 &&
          Object.entries(environmentVars).map(([dictKey, dictValue]) => (
            <Row key={dictKey}>
              <Col style={{ flex: 0.1 }}></Col>
              <Col style={{ flex: 2.2 }}>
                <em>{`${dictKey}: `}</em>
              </Col>
              <Col style={{ flex: 12.5 }}>
                <FieldBadge field={dictValue} color={'#7e7c7c'} />
              </Col>
            </Row>
          ))}
      </Row>
    </Fragment>
  );
};

const updateEditEnvironmentVariables = (newEnvs, initialEnvs, setEnvironmentVariables, setRequestEnvironmentVars) => {
  const validEnvironmentVariables = {};
  newEnvs.map((variable) => {
    // a valid key and value must be set for each uploaded environment variable
    if (variable['key']) {
      validEnvironmentVariables[variable['key']] = variable['value'] == '' ? null : variable['value'];
    }
  });

  const requestEnvironmentVariables = { add_env: validEnvironmentVariables };
  if (initialEnvs && Object.keys(initialEnvs).length) {
    requestEnvironmentVariables['remove_env'] = Object.keys(initialEnvs);
  }
  setRequestEnvironmentVars(requestEnvironmentVariables);
  setEnvironmentVariables(newEnvs);
};

const updateCreateEnvironmentVariables = (newEnvs, setEnvironmentVariables, setRequestEnvironmentVars) => {
  setRequestEnvironmentVars(newEnvs);
  setEnvironmentVariables(newEnvs);
};

const EnvironmentVars = ({ environmentVars, setRequestEnvironmentVars, mode }) => {
  const [envVars, setEnvVars] = useState(
    Object.keys(environmentVars).length
      ? Object.keys(environmentVars).map((item) => {
          return { key: item, value: environmentVars[item] };
        })
      : [{ key: '', value: '' }],
  );

  const updateEnvs = (newEnvs) => {
    if (mode == 'Edit') {
      return updateEditEnvironmentVariables(newEnvs, environmentVars, setEnvVars, setRequestEnvironmentVars);
    }
    return updateCreateEnvironmentVariables(newEnvs, setEnvVars, setRequestEnvironmentVars);
  };

  // this is needed for onload when cloning from an existing image
  useEffect(() => {
    // we don't check for errors because we assume the source for copy is valid and
    // might want to not assume that
    setRequestEnvironmentVars(envVars);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Fragment>
      {mode == 'Edit' && (
        <Fragment>
          <Row>
            <Col style={{ flex: 2.5 }}>
              <OverlayTipRight tip={EnvironmentToolTip}>
                <b>{'Environment'}</b> <FaQuestionCircle />
              </OverlayTipRight>
            </Col>
            <Col style={{ flex: 12.5 }} />
          </Row>
          <Row>
            <Col style={{ flex: 0.1 }} />
            <Col style={{ flex: 2.2 }} />
            <Col style={{ flex: 12.5 }}>
              <div className="image-fields">
                <SelectableDictionary
                  entries={envVars}
                  disabled={false}
                  setEntries={updateEnvs}
                  keyPlaceholder={'New Variable'}
                  valuePlaceholder={'New Value'}
                  trim={true}
                />
              </div>
            </Col>
          </Row>
        </Fragment>
      )}
      {mode != 'Edit' && (
        <Row>
          <Col className="title-col">
            <h5>Environment</h5>
          </Col>
          <Col className="field-col">
            <OverlayTipRight tip={EnvironmentToolTip}>
              <SelectableDictionary
                entries={envVars}
                disabled={false}
                setEntries={updateEnvs}
                keyPlaceholder={'New Variable'}
                valuePlaceholder={'New Value'}
                trim={true}
              />
            </OverlayTipRight>
          </Col>
        </Row>
      )}
    </Fragment>
  );
};

const ImageEnvironmentVariables = ({ environmentVars, setRequestEnvironmentVars, mode }) => {
  if (mode == 'View') {
    return <DisplayEnvironmentVars environmentVars={environmentVars} />;
  }
  return <EnvironmentVars environmentVars={environmentVars} setRequestEnvironmentVars={setRequestEnvironmentVars} mode={mode} />;
};

export default ImageEnvironmentVariables;
