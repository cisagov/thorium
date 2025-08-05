import React, { Fragment, useEffect, useState } from 'react';
import { Alert, Col, Form, Row } from 'react-bootstrap';
import { FaQuestionCircle } from 'react-icons/fa';

// project imports
import { FieldBadge, OverlayTipRight, SelectableArray, Subtitle } from '@components';

const ArgumentToolTips = {
  self: `The command line parameters passed to this image when it is run.`,
  entrypoint: `The entrypoint executable or script that the agent will call to run this image. 
    For images run in K8s,  leaving this blank will cause the default container entrypoint to be 
    used.`,
  command: `The command arguments to pass the entrypoint of the image.`,
  reaction: `The flag used to pass in the reaction ID of the running image.`,
  repo: `The flag used to pass in a git repo. This is only used by Thorium data generation jobs.`,
  commit: `The flag used to pass in the specific repo commit. This is only used by Thorium data 
    generation jobs.`,
  output: `The flag or arg position used to pass in an output path for this image's results.`,
  kwarg: `The actual flag used to pass in the output path.`,
};

const ArgumentsTemplate = {
  entrypoint: [],
  command: [],
  reaction: '',
  repo: '',
  commit: '',
  output: 'None',
  kwarg: '',
};

const OutputTypes = ['None', 'Append', 'Kwarg'];

const DisplayImageArguments = ({ args }) => {
  return (
    <Fragment>
      {/** ********************************************** Entry Point */}
      <Row>
        <Col className="key-col-1"></Col>
        <Col className="key-col-2-ext">
          <em>{`entrypoint: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ArgumentToolTips.entrypoint}>
              <FieldBadge field={args.entrypoint} color={'#7e7c7c'} />
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
      {/** ********************************************** Command */}
      <Row>
        <Col className="key-col-1"></Col>
        <Col className="key-col-2-ext">
          <em>{`command: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ArgumentToolTips.command}>
              <FieldBadge field={args.command} color={'default'} />
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
      {/** ********************************************** Reaction */}
      <Row>
        <Col className="key-col-1"></Col>
        <Col className="key-col-2-ext">
          <em>{`reaction: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ArgumentToolTips.reaction}>
              <FieldBadge field={args.reaction} color={'#7e7c7c'} />
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
      {/** ********************************************** Repo */}
      <Row>
        <Col className="key-col-1"></Col>
        <Col className="key-col-2-ext">
          <em>{`repo: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ArgumentToolTips.repo}>
              <FieldBadge field={args.repo} color={'#7e7c7c'} />
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
      {/** ********************************************** Commit */}
      <Row>
        <Col className="key-col-1"></Col>
        <Col className="key-col-2-ext">
          <em>{`commit: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ArgumentToolTips.commit}>
              <FieldBadge field={args.commit} color={'#7e7c7c'} />
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
      {/** ********************************************** Output */}
      <Row>
        <Col className="key-col-1"></Col>
        <Col className="key-col-2-ext">
          <em>{`output: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ArgumentToolTips.output}>
              <FieldBadge field={args.output} color={'#7e7c7c'} />
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
    </Fragment>
  );
};

const updateCreateRequest = (args, setRequestArguments, setErrors, setHasErrors) => {
  const requestArguments = structuredClone(args);
  const errors = {};

  // if this is uninitialized in some way short circuit
  if (!requestArguments) return;

  if (
    !Array.isArray(requestArguments.entrypoint) ||
    (requestArguments.entrypoint.length == 1 && requestArguments.entrypoint[0] == '') ||
    requestArguments.entrypoint.length == 0
  ) {
    delete requestArguments.entrypoint;
  }
  if (
    !Array.isArray(requestArguments.command) ||
    (requestArguments.command.length == 1 && requestArguments.command[0] == '') ||
    requestArguments.command.length == 0
  ) {
    delete requestArguments.command;
  }
  if (requestArguments.reaction == '') {
    delete requestArguments.reaction;
  }
  if (requestArguments.repo == '') {
    delete requestArguments.repo;
    delete requestArguments.commit;
  }
  if (requestArguments.commit == '') {
    delete requestArguments.commit;
  }
  if (requestArguments.output == 'Kwarg' && requestArguments.kwarg == '') {
    delete requestArguments.output;
    errors['output'] = `Kwarg flag must be specified when 'Kwarg' is selected for output`;
  } else if (requestArguments.output == 'Kwarg') {
    requestArguments['output'] = { Kwarg: requestArguments.kwarg };
  }
  // kwarg isn't a request field, it just makes dealing with the 'output' field easier
  // otherwise output would be output: {'Kwarg': '--flag'}
  if (requestArguments.kwarg) delete requestArguments.kwarg;
  // update displayed errors
  setErrors(errors);
  // tell parent there are unresolved errors
  Object.keys(errors).length > 0 ? setHasErrors(true) : setHasErrors(false);
  // update the parents request args structure
  setRequestArguments(requestArguments);
};

const updateEditRequest = (args, setRequestArguments, setErrors, setHasErrors) => {
  const requestArguments = structuredClone(args);
  const errors = {};

  // if this is uninitialized in some way short circuit
  if (!requestArguments) return;

  if (
    !Array.isArray(requestArguments.entrypoint) ||
    (requestArguments.entrypoint.length == 1 && requestArguments.entrypoint[0] == '') ||
    requestArguments.entrypoint.length == 0
  ) {
    delete requestArguments.entrypoint;
    requestArguments['clear_entrypoint'] = true;
  }
  if (
    !Array.isArray(requestArguments.command) ||
    (requestArguments.command.length == 1 && requestArguments.command[0] == '') ||
    requestArguments.command.length == 0
  ) {
    delete requestArguments.command;
    requestArguments['clear_command'] = true;
  }
  if (requestArguments.reaction == null || requestArguments.reaction == '') {
    delete requestArguments.reaction;
    requestArguments['clear_reaction'] = true;
  }
  if (requestArguments.repo == null || requestArguments.repo == '') {
    delete requestArguments.repo;
    delete requestArguments.commit;
    requestArguments['clear_repo'] = true;
    requestArguments['clear_commit'] = true;
  }
  if (requestArguments.commit == null || requestArguments.commit == '') {
    delete requestArguments.commit;
    requestArguments['clear_commit'] = true;
  }
  if (requestArguments.output == 'Kwarg') {
    if (requestArguments.kwarg == null || requestArguments.kwarg == '') {
      delete requestArguments.output;
      delete requestArguments.kwarg;
      // this is an error state since Kwarg requires a flag
      errors['output'] = `Kwarg flag must be specified when 'Kwarg' is selected for 
        output argument.`;
    } else {
      // this must be output = {'Kwarg': '--some-flag}
      requestArguments.output = { Kwarg: requestArguments.kwarg };
    }
  }
  // kwarg isn't a request field, it just makes dealing with the 'output' field easier
  // otherwise output would be output: {'Kwarg': '--flag'}
  if (requestArguments.kwarg) delete requestArguments.kwarg;
  setErrors(errors);
  // tell parent there are unresolved errors
  Object.keys(errors).length > 0 ? setHasErrors(true) : setHasErrors(false);
  // update the parents request args structure
  setRequestArguments(requestArguments);
};

const ArgumentFields = ({ initialArgs, setRequestArguments, errors }) => {
  if (initialArgs && initialArgs.output && Object.keys(initialArgs.output).includes('Kwarg')) {
    initialArgs.kwarg = initialArgs.output['Kwarg'];
    initialArgs.output = 'Kwarg';
  }
  const [args, setArgs] = useState(structuredClone(initialArgs));

  // update a argument value for key
  const updateArguments = (key, value) => {
    // make a deep copy of the argument
    const argsCopy = structuredClone(args);
    // set the new value for the key
    argsCopy[key] = value;
    // update the argument object and trigger dom refresh
    setArgs(argsCopy);
    setRequestArguments(argsCopy);
  };

  // this is needed for onload when cloning from an existing image
  useEffect(() => {
    setRequestArguments(initialArgs);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="image-fields">
      <OverlayTipRight tip={ArgumentToolTips.entrypoint}>
        <Form.Group>
          <Form.Label>
            <Subtitle>Entry Point</Subtitle>
          </Form.Label>
          <SelectableArray
            initialEntries={args && args.entrypoint && Array.isArray(args.entrypoint) ? args.entrypoint.map((item) => item.trim()) : ['']}
            setEntries={(entryArgs) => updateArguments('entrypoint', entryArgs)}
            placeholder={'entry point'}
            trim={true}
            disabled={false}
          />
        </Form.Group>
      </OverlayTipRight>
      {/** ********************************************** Command */}
      <OverlayTipRight tip={ArgumentToolTips.command}>
        <Form.Group>
          <Form.Label>
            <Subtitle>Command</Subtitle>
          </Form.Label>
          <SelectableArray
            initialEntries={args && args.command && Array.isArray(args.command) ? args.command.map((item) => item.trim()) : ['']}
            setEntries={(entryCmds) => updateArguments('command', entryCmds)}
            placeholder={'command'}
            trim={true}
            disabled={false}
          />
        </Form.Group>
      </OverlayTipRight>
      {/** ********************************************** Reaction */}
      <OverlayTipRight tip={ArgumentToolTips.reaction}>
        <Form.Group className="mb-2">
          <Form.Label>
            <Subtitle>Reaction</Subtitle>
          </Form.Label>
          <Form.Control
            type="text"
            value={args && args.reaction ? args.reaction : ''}
            placeholder="reaction"
            onChange={(e) => updateArguments('reaction', String(e.target.value).trim())}
          />
        </Form.Group>
      </OverlayTipRight>
      {/** ********************************************** Repo */}
      <OverlayTipRight tip={ArgumentToolTips.repo}>
        <Form.Group className="mb-2">
          <Form.Label>
            <Subtitle>Repo</Subtitle>
          </Form.Label>
          <Form.Control
            type="text"
            value={args && args.repo ? args.repo : ''}
            placeholder="repo"
            onChange={(e) => updateArguments('repo', String(e.target.value).trim())}
          />
        </Form.Group>
      </OverlayTipRight>
      {/** ********************************************** Commit */}
      <OverlayTipRight tip={ArgumentToolTips.commit}>
        <Form.Group className="mb-2">
          <Form.Label>
            <Subtitle>Commit</Subtitle>
          </Form.Label>
          <Form.Control
            type="text"
            value={args && args.commit ? args.commit : ''}
            placeholder="commit"
            onChange={(e) => updateArguments('commit', String(e.target.value).trim())}
          />
        </Form.Group>
      </OverlayTipRight>
      {/** ********************************************** Output */}
      <OverlayTipRight tip={ArgumentToolTips.output}>
        <Form.Group className="mb-2">
          <Form.Label>
            <Subtitle>Output</Subtitle>
          </Form.Label>
          <Form.Select value={args.output} onChange={(e) => updateArguments('output', String(e.target.value).trim())}>
            {OutputTypes.sort().map((output) => (
              <option key={output} value={output}>
                {output}
              </option>
            ))}
          </Form.Select>
        </Form.Group>
      </OverlayTipRight>
      {args && args.output == 'Kwarg' && (
        <OverlayTipRight tip={ArgumentToolTips.kwarg}>
          <Form.Group className="mb-2 image-fields">
            <Form.Label>
              <Subtitle>Kwarg</Subtitle>
            </Form.Label>
            <Form.Control
              type="text"
              value={args && args.kwarg ? args.kwarg : ''}
              placeholder="kwarg option"
              onChange={(e) => updateArguments('kwarg', String(e.target.value).trim())}
            ></Form.Control>
          </Form.Group>
        </OverlayTipRight>
      )}
      {errors && 'output' in errors && (
        <Alert variant="danger" className="d-flex justify-content-center m-2">
          {errors.output}
        </Alert>
      )}
    </div>
  );
};

const EditImageArguments = ({ args, setRequestArguments, errors }) => {
  return (
    <Row>
      <Col style={{ flex: 0.2 }}></Col>
      <Col style={{ flex: 1.25 }}></Col>
      <Col style={{ flex: 8 }}>
        <ArgumentFields initialArgs={args} errors={errors} setRequestArguments={setRequestArguments} />
      </Col>
    </Row>
  );
};

const CreateImageArguments = ({ args, setRequestArguments, errors }) => {
  return (
    <Row>
      <Col className="title-col">
        <h5>Arguments</h5>
      </Col>
      <Col className="field-col">
        <ArgumentFields initialArgs={args} errors={errors} setRequestArguments={setRequestArguments} />
      </Col>
    </Row>
  );
};

const ImageArguments = ({ args, setRequestArguments, setHasErrors, mode }) => {
  const [errors, setErrors] = useState({});
  // provide the edit/create components with a callback to update a
  // request formatted args object
  const setUpdatedRequestArguments = (newArguments) => {
    if (['Create', 'Copy'].includes(mode)) {
      return updateCreateRequest(newArguments, setRequestArguments, setErrors, setHasErrors);
    } else {
      return updateEditRequest(newArguments, setRequestArguments, setErrors, setHasErrors);
    }
  };

  if (mode == 'Copy') {
    return <CreateImageArguments args={args} errors={errors} setRequestArguments={setUpdatedRequestArguments} />;
  } else if (mode == 'Create') {
    return <CreateImageArguments args={ArgumentsTemplate} errors={errors} setRequestArguments={setUpdatedRequestArguments} />;
  }
  return (
    <Fragment>
      <Row>
        <Col>
          <OverlayTipRight tip={ArgumentToolTips.self}>
            <b>{'Arguments'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
      </Row>
      {mode == 'View' && <DisplayImageArguments args={args} />}
      {mode == 'Edit' && <EditImageArguments args={args} errors={errors} setRequestArguments={setUpdatedRequestArguments} />}
    </Fragment>
  );
};

export default ImageArguments;
