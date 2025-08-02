import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Button, Row, Col, Alert } from 'react-bootstrap';
import { FaAngleDown, FaAngleUp } from 'react-icons/fa';

// project imports
import {
  OverlayTipRight,
  ImageFields,
  ImageArguments,
  ImageDependencies,
  ImageResources,
  ImageEnvironmentVariables,
  ImageVolumes,
  ImageNetworkPolicies,
  ImageSecurityContext,
  ImageOutputCollection,
  LoadingSpinner,
  Page,
} from '@components';
import { useAuth, fetchGroups, fetchImages } from '@utilities';
import { createImage } from '@thorpi';

const CreateImageContainer = () => {
  const [hideAdvanced, setHideAdvanced] = useState(true);
  // Image set state functions
  const [groups, setGroups] = useState([]);
  const [images, setImages] = useState([]);
  const [imageFields, setImageFields] = useState({});
  const [volumes, setVolumes] = useState({}); // optional
  const [environmentVars, setEnvironmentVars] = useState([{ key: '', value: '' }]); // optional
  const [securityContext, setSecurityContext] = useState({}); // optional
  const [resources, setResources] = useState({}); // optional
  const [args, setArgs] = useState({}); // optional
  const [argErrors, setArgErrors] = useState(false);
  const [dependencies, setDependencies] = useState({});
  const [outputCollection, setOutputCollection] = useState({});
  const [networkPolicies, setNetworkPolicies] = useState([]); // optional
  // Set error state functions
  const [displayErrors, setDisplayErrors] = useState(false);
  // required fields are blank at start
  const [imageFieldErrors, setImageFieldErrors] = useState(true); // this is true at start
  const [resourceErrors, setResourceErrors] = useState(false);
  const [createImageErrors, setCreateImageErrors] = useState('');
  const [dependencyErrors, setDependencyErrors] = useState(false);
  const [volumeErrors, setVolumeErrors] = useState(false);
  const [outputCollectionErrors, setOutputCollectionErrors] = useState(false);
  const navigate = useNavigate();
  const { state } = useLocation();
  const { userInfo, checkCookie } = useAuth();
  const [loading, setLoading] = useState(false);
  let cancelUpdate = false;

  // need user's group roles to validate permissions to create/edit/delete pipelines
  useEffect(() => {
    fetchGroups(setGroups, null, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // get list of images for the selected group for use in dependencies drop downs
  useEffect(() => {
    // if state is passed, use that
    const group = state && state.group ? state.group : imageFields.group;
    if (group) fetchImages([group], setImages, cancelUpdate, checkCookie, setLoading, false);
    return () => {
      // eslint-disable-next-line react-hooks/exhaustive-deps
      cancelUpdate = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageFields.group]);

  // clear the create error if all others have been resolved
  useEffect(() => {
    if (!(imageFieldErrors || volumeErrors || dependencyErrors || outputCollectionErrors || resourceErrors)) {
      setCreateImageErrors('');
    }
  }, [imageFieldErrors, volumeErrors, dependencyErrors, outputCollectionErrors, resourceErrors]);

  /**
   * Create a new Thorium image
   * @returns {void}
   */
  async function handleImageCreate() {
    let data = {};

    // set image fields
    if (Object.keys(imageFields).length) {
      data = structuredClone(imageFields);
    }

    // ------------------- resources --------------------
    if (Object.keys(resources).length && data.scaler != 'External') {
      data['resources'] = resources;
    }

    // ------------------- network policies -------------
    if (Object.keys(networkPolicies).length && data.scaler == 'K8s') {
      data['network_policies'] = networkPolicies;
    }

    // ------------------- arguments --------------------
    if (Object.keys(args).length) {
      data['args'] = args;
    }

    // ------------------------ volumes ----------------------------
    if (volumes && volumes.length && data.scaler == 'K8s') {
      data['volumes'] = volumes;
    }

    // ------------------------ output collection ----------------------------
    if (Object.keys(outputCollection).length && data.scaler != 'External') {
      data['output_collection'] = outputCollection;
    }

    // ------------------------ dependencies ----------------------------
    if (Object.keys(dependencies).length) {
      data['dependencies'] = dependencies;
    }

    // -------------------- security_context -----------------------
    // Only Admins can set security_context values, so don't include request for other roles
    if (Object.keys(securityContext).length && userInfo && userInfo.role == 'Admin' && data.scaler != 'External') {
      data['security_context'] = securityContext;
    }
    // --------------------------- tags ----------------------------
    const environmentVarsJson = {};
    // tags are key/value pairs where the values are strings
    if (environmentVars) {
      // convert envVariables list into a dictionary of env key/value pairs
      environmentVars.map((variable) => {
        // a valid key and value must be set for each uploaded tag
        if (variable['key']) {
          environmentVarsJson[variable['key']] = variable['value'] == '' ? null : variable['value'];
        }
      });

      // only append environment variables json if it is not empty
      if (Object.keys(environmentVarsJson).length && (data.scaler == 'K8s' || data.scaler == 'BareMetal')) {
        data['env'] = environmentVarsJson;
      }
    }

    // check if there are fields that are missing or invalid
    // this is used to put an alert by the create button
    // so that users aren't confused why their image wasn't created
    // when there are errors but the optionals are expanded so they
    // don't see the errors at the top of the page
    if (imageFieldErrors || resourceErrors || argErrors || outputCollectionErrors || dependencyErrors || volumeErrors) {
      setCreateImageErrors('Please resolve missing fields or invalid entries');
      setDisplayErrors(true);
      return;
    } else {
      // this is really only needed if we aren't redirecting on success
      setCreateImageErrors('');
    }

    // redirect back to the images display page if the image was created successfully
    if (await createImage(data, setCreateImageErrors)) {
      navigate('/images');
    } else {
      // if a user makes an invalid change to the form field after an initial request failure,
      // display those error alerts
      setDisplayErrors(true);
    }
  }

  return (
    <Page className="image-create" title="Create Image">
      <Row>
        <center>
          <h3>Create An Image</h3>
        </center>
      </Row>
      <Row className="mt-4">
        <Col className="title-col">
          <h5>Image</h5>
        </Col>
        <Col className="field-col">
          <ImageFields
            image={state ? state : imageFields}
            groups={groups ? groups : []}
            setRequestImageFields={setImageFields}
            setHasErrors={setImageFieldErrors}
            showErrors={state ? true : displayErrors}
            mode={state ? 'Copy' : 'Create'}
          />
        </Col>
      </Row>
      <Row>
        <Col className="d-flex justify-content-center">
          <OverlayTipRight tip={`${hideAdvanced ? 'Expand' : 'Hide'} optional fields`}>
            <div className="icon-btn" onClick={() => setHideAdvanced(!hideAdvanced)}>
              {hideAdvanced ? <FaAngleDown size="36" /> : <FaAngleUp size="36" />}
            </div>
          </OverlayTipRight>
        </Col>
      </Row>
      <hr className="mt-0" />
      <div className={`${hideAdvanced ? 'advanced-hidden' : ''}`}>
        <ImageResources
          resources={state && state.resources ? state.resources : {}}
          setRequestResources={setResources}
          setHasErrors={setResourceErrors}
          mode={state ? 'Copy' : 'Create'}
        />
        <hr />
        <ImageArguments
          args={state && state.args ? state.args : {}}
          setRequestArguments={setArgs}
          setHasErrors={setArgErrors}
          mode={state ? 'Copy' : 'Create'}
        />
        <hr />
        <ImageOutputCollection
          outputCollection={state && state.output_collection ? state.output_collection : {}}
          setRequestOutputCollection={setOutputCollection}
          groups={userInfo && userInfo.groups ? userInfo.groups : []}
          mode={state ? 'Copy' : 'Create'}
          setHasErrors={setOutputCollectionErrors}
          disabled={imageFields['scaler'] && imageFields.scaler == 'External'}
        />
        <hr />
        <ImageDependencies
          images={images}
          dependencies={state && state.dependencies ? state.dependencies : {}}
          setErrors={setDependencyErrors}
          setRequestDependencies={setDependencies}
          mode={state ? 'Copy' : 'Create'}
          disabled={imageFields['scaler'] && imageFields.scaler == 'External'}
        />
        <hr />
        <ImageEnvironmentVariables
          environmentVars={state && state.env ? state.env : {}}
          setRequestEnvironmentVars={setEnvironmentVars}
          mode={state ? 'Copy' : 'Create'}
        />
        <hr />
        <ImageVolumes
          volumes={state && state.volumes ? state.volumes : []}
          setRequestVolumes={setVolumes}
          mode={state ? 'Copy' : 'Create'}
          setHasErrors={setVolumeErrors}
          disabled={imageFields['scaler'] && imageFields.scaler != 'K8s'}
        />
        <hr />
        <ImageNetworkPolicies
          policies={state && state.network_policies ? state.network_policies : {}}
          setRequestNetworkPolicies={setNetworkPolicies}
          mode={state ? 'Copy' : 'Create'}
        />
        <hr />
        <ImageSecurityContext
          securityContext={state && state.security_context ? state.security_context : {}}
          setRequestSecurityContext={setSecurityContext}
          mode={state ? 'Copy' : 'Create'}
          disabled={(imageFields['scaler'] && imageFields.scaler == 'External') || !userInfo || userInfo.role != 'Admin'}
        />
      </div>
      <Row className="d-flex justify-content-center">
        <Col>
          {createImageErrors && (
            <Alert variant="danger" className="d-flex justify-content-center m-2">
              {createImageErrors}
            </Alert>
          )}
        </Col>
      </Row>
      <Row>
        <LoadingSpinner loading={loading}></LoadingSpinner>
      </Row>
      <Row className="mt-3">
        <Col className="d-flex justify-content-center">
          <Button className="secondary-btn" onClick={() => navigate(-1)}>
            Cancel
          </Button>
          <Button className="ok-btn" onClick={() => handleImageCreate()}>
            Create
          </Button>
        </Col>
      </Row>
    </Page>
  );
};

export default CreateImageContainer;
