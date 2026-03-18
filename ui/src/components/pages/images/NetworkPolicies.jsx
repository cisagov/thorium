import { Fragment, useEffect, useState } from 'react';
import { Col, Row } from 'react-bootstrap';
import { FaQuestionCircle } from 'react-icons/fa';

// project imports
import FieldBadge from '@components/shared/badges/FieldBadge';
import SelectableArray from '@components/shared/selectable/SelectableArray';
import { OverlayTipRight } from '@components/shared/overlay/tips';

const NetworkPolicyToolTip = `Network policies dictate which entities the image can connect to or receive
    communication from.`;

const updateEditNetworkPolicies = (newPolicies, initialPolicies, setNetworkPolicies, setRequestNetworkPolicies) => {
  const policiesRemoved = initialPolicies.filter((policy) => {
    return !newPolicies.includes(policy);
  });
  const policiesAdded = newPolicies.filter((policy) => {
    return !initialPolicies.includes(policy);
  });

  const requestNetworkPolicies = {
    policies_added: policiesAdded,
    policies_removed: policiesRemoved,
  };
  setRequestNetworkPolicies(requestNetworkPolicies);
  setNetworkPolicies(newPolicies);
};

const updateCreateNetworkPolicies = (newPolicies, setNetworkPolicies, setRequestNetworkPolicies) => {
  setRequestNetworkPolicies(newPolicies);
  setNetworkPolicies(newPolicies);
};

const NetworkPoliciesEdit = ({ initialPolicies, setRequestNetworkPolicies, mode }) => {
  const [policies, setNetworkPolicies] = useState(structuredClone(initialPolicies));

  const updatePolicies = (newPolicies) => {
    if (mode == 'Edit') {
      return updateEditNetworkPolicies(newPolicies, initialPolicies, setNetworkPolicies, setRequestNetworkPolicies);
    }
    return updateCreateNetworkPolicies(newPolicies, setNetworkPolicies, setRequestNetworkPolicies);
  };

  // this is needed for onload when cloning from an existing image
  useEffect(() => {
    // don't check for errors and assume initial policies are valid
    setRequestNetworkPolicies(initialPolicies);
  }, []);

  return (
    <Fragment>
      {mode == 'Edit' && (
        <Fragment>
          <Row>
            <Col style={{ flex: 2.5 }}>
              <OverlayTipRight tip={NetworkPolicyToolTip}>
                <b>{'Network Policies'}</b> <FaQuestionCircle />
              </OverlayTipRight>
            </Col>
          </Row>
          <Row>
            <Col style={{ flex: 0.2 }}></Col>
            <Col style={{ flex: 1.25 }}></Col>
            <Col style={{ flex: 8 }}>
              <SelectableArray
                initialEntries={policies}
                setEntries={updatePolicies}
                disabled={false}
                placeholder={'example-policy'}
                trim={true}
              />
            </Col>
          </Row>
        </Fragment>
      )}
      {mode != 'Edit' && (
        <Row>
          <Col className="title-col">
            <h5>Network Policies</h5>
          </Col>
          <Col className="field-col">
            <OverlayTipRight tip={NetworkPolicyToolTip}>
              <SelectableArray
                initialEntries={initialPolicies}
                setEntries={updatePolicies}
                disabled={false}
                placeholder={'example-policy'}
                trim={true}
              />
            </OverlayTipRight>
          </Col>
        </Row>
      )}
    </Fragment>
  );
};

const DisplayImageNetworkPolicies = ({ policies }) => {
  return (
    <Fragment>
      <Row>
        <Col style={{ flex: 2.5 }}>
          <OverlayTipRight tip={NetworkPolicyToolTip}>
            <b>{'Network Policies'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
        <Col style={{ flex: 12.5 }}>
          <FieldBadge field={policies} color={'#7e7c7c'} />
        </Col>
      </Row>
    </Fragment>
  );
};

const NetworkPolicies = ({ policies, setRequestNetworkPolicies, mode }) => {
  if (mode == 'View') {
    return <DisplayImageNetworkPolicies policies={policies} />;
  }
  return <NetworkPoliciesEdit initialPolicies={policies} setRequestNetworkPolicies={setRequestNetworkPolicies} mode={mode} />;
};

export default NetworkPolicies;
