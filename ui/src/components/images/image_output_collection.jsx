import React, { Fragment, useEffect, useState } from 'react';
import { Alert, Badge, Col, Form, Row } from 'react-bootstrap';
import { FaQuestionCircle } from 'react-icons/fa';

// project imports
import { FieldBadge, OverlayTipRight, SelectableArray, SelectableDictionary, SelectGroups, Subtitle } from '@components';

const OutputCollectionToolTips = {
  self: `Configurations that determine how the Thorium agent will intake analysis 
    artifacts after running this image.`,
  files: {
    names: `Names of specific result files to collect. The default behavior collects 
      all files in the result files directory.`,
    tags: `Path to a JSON formatted file of key/value tags.`,
    results: `Path to a single renderable results file (JSON, Table, String, etc).`,
    result_files: `Path to a directory containing one or more result file(s). These 
      are displayed as download links.`,
  },
  groups: `Limit this image to uploading results to these selected groups. By default 
    results are uploaded to the group(s) of the file or repo that the image ran on.`,
  children: `Path to children samples extracted by the image.`,
  auto_tag: `Specific keys that will get automatically added as tags from the image 
    results. JSON formatted results are required to use auto tagging.`,
};

const OutputCollectionTemplate = {
  files: {
    names: '',
    tags: '',
    results: '',
    result_files: '',
  },
  select_groups: {},
  groups: [],
  children: '',
  auto_tag: [],
  select_auto_tag: [{ key: '', value: '' }],
};

const DisplayOutputCollection = ({ outputCollection }) => {
  return (
    <Fragment>
      {outputCollection && Object.keys(outputCollection).length > 0 && (
        <Fragment>
          <Row>
            <Col style={{ flex: 0.1 }}></Col>
            <Col style={{ flex: 2.2 }}>
              <em>{`files`}</em>
            </Col>
            <Col style={{ flex: 12.5 }}></Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`results`}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <FieldBadge field={outputCollection['files']['results']} color={'#7e7c7c'} />
            </Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`result_files`}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <FieldBadge field={outputCollection['files']['result_files']} color={'#7e7c7c'} />
            </Col>
          </Row>
          {outputCollection['files']['names'].length > 0 &&
            outputCollection['files']['names'].length == 1 &&
            outputCollection['files']['names'][0] != '' && (
              <Row>
                <Col style={{ flex: 1 }}></Col>
                <Col style={{ flex: 3 }}>
                  <em>{`file_names`}</em>
                </Col>
                <Col style={{ flex: 21.5 }}>
                  <FieldBadge field={outputCollection['files']['names']} color={'#7e7c7c'} />
                </Col>
              </Row>
            )}
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`tags`}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <FieldBadge field={outputCollection['files']['tags']} color={'#7e7c7c'} />
            </Col>
          </Row>
          <Row>
            <Col style={{ flex: 0.1 }}></Col>
            <Col style={{ flex: 2.2 }}>
              <em>{`children`}</em>
            </Col>
            <Col style={{ flex: 12.5 }}>
              <FieldBadge field={outputCollection['children']} color={'#7e7c7c'} />
            </Col>
          </Row>
          {Object.entries(outputCollection['auto_tag']).length > 0 && (
            <Row>
              <Col style={{ flex: 0.1 }}></Col>
              <Col style={{ flex: 2.2 }}>
                <em>{`auto tagging: `}</em>
              </Col>
              <Col style={{ flex: 12.5 }}>
                {Object.entries(outputCollection['auto_tag']).map((tag, i) => {
                  if (tag.length) {
                    return (
                      <Badge key={i} style={{ backgroundColor: '#7e7c7c' }}>
                        {`${tag[0]}: ${tag[1]['key']}`}
                      </Badge>
                    );
                  }
                })}
              </Col>
            </Row>
          )}
          {outputCollection['groups'].length > 0 && (
            <Row>
              <Col style={{ flex: 0.1 }}></Col>
              <Col style={{ flex: 2.2 }}>
                <em>{`groups: `}</em>
              </Col>
              <Col style={{ flex: 12.5 }}>
                <FieldBadge field={outputCollection['groups']} color={'#7e7c7c'} />
              </Col>
            </Row>
          )}
        </Fragment>
      )}
    </Fragment>
  );
};

const updateCreateRequestOutputCollection = (newOutputCollection, setRequestOutputCollection, setErrors, setHasErrors) => {
  const requestOutputCollection = structuredClone(newOutputCollection);
  const errors = {};

  // do not pass back blank or null values for string fields
  if (newOutputCollection.files.results == '' || newOutputCollection.files.results == null) {
    delete requestOutputCollection.files.results;
  }
  if (newOutputCollection.files.result_files == '' || newOutputCollection.files.result_files == null) {
    delete requestOutputCollection.files.result_files;
  }
  if (newOutputCollection.files.names == '' || newOutputCollection.files.names == null) {
    delete requestOutputCollection.files.names;
  }
  if (newOutputCollection.files.tags == '' || newOutputCollection.files.tags == null) {
    delete requestOutputCollection.files.tags;
  }
  if (newOutputCollection.children == '' || newOutputCollection.children == null) {
    delete requestOutputCollection.children;
  }
  if (Object.keys(requestOutputCollection.files).length == 0) {
    delete requestOutputCollection.files;
  }

  // change 'auto_tag' config to map to Thorium format
  const tags = {};
  if (requestOutputCollection.select_auto_tag) {
    requestOutputCollection.select_auto_tag.forEach((tag) => {
      // tag is empty but rename of key for tag has data entered
      if (tag['key'].trim() == '' && tag['value'].trim().length > 0) {
        errors['auto_tag'] = 'Tag Names Cannot Be Empty';
        setHasErrors(true);
      }

      // only add tag if tag data exists in array
      if (tag['key'].trim().length > 0) {
        // key is tag you want to autotag, value is new name of tag's key
        tags[tag['key']] = {
          logic: 'Exists',
          key: tag['value'].trim() == '' ? null : tag['value'],
        };
      }
    });
  }
  requestOutputCollection.auto_tag = tags;

  // change 'groups' config to map to Thorium format
  const requestGroups = [];
  Object.keys(requestOutputCollection.select_groups).map((group) => {
    if (requestOutputCollection.select_groups[group]) {
      requestGroups.push(group);
    }
  });
  if (requestGroups.length > 0) {
    requestOutputCollection.groups = requestGroups;
  } else {
    delete requestOutputCollection.groups;
  }

  // clean up local structures used to map image info to renderable components
  delete requestOutputCollection.select_auto_tag;
  delete requestOutputCollection.select_groups;

  // check for errors, notify parent if they exist
  if (Object.keys(errors).length > 0) {
    setHasErrors(true);
  } else {
    setHasErrors(false);
  }
  setErrors(errors);
  setRequestOutputCollection(requestOutputCollection);
};

const updateEditRequestOutputCollection = (
  initialOutputCollection,
  newOutputCollection,
  setRequestOutputCollection,
  setErrors,
  setHasErrors,
) => {
  const requestOutputCollection = structuredClone(newOutputCollection);
  const errors = {};

  // clear files configs if they have all be removed from the form
  if (
    newOutputCollection.files.results == '' &&
    newOutputCollection.files.result_files == '' &&
    newOutputCollection.files.names == '' &&
    newOutputCollection.files.tags == ''
  ) {
    // now check if there are any 'files' keys left, if not lets set to clear
    // we do this because empty strings mean clear
    if (Object.keys(requestOutputCollection.files).length == 0) {
      delete requestOutputCollection.files;
      requestOutputCollection['clear_files'] = true;
    }
  }

  // file names vector must be cleared before being patched
  if (requestOutputCollection['files'] && requestOutputCollection.files['names']) {
    requestOutputCollection.files['clear_names'] = true;
    requestOutputCollection.files['add_names'] = [...requestOutputCollection.files.names];
    delete requestOutputCollection.files.names;
  }

  // change 'groups' config to map to Thorium format
  const requestGroups = [];
  Object.keys(requestOutputCollection.select_groups).map((group) => {
    if (requestOutputCollection.select_groups[group]) {
      requestGroups.push(group);
    }
  });
  if (requestGroups.length) {
    requestOutputCollection.groups = requestGroups;
  } else {
    delete requestOutputCollection.groups;
    requestOutputCollection['clear_groups'] = true;
  }

  // change 'auto_tag' config to map to Thorium format
  const tags = {};
  if (requestOutputCollection.select_auto_tag) {
    requestOutputCollection.select_auto_tag.forEach((tag) => {
      // tag is empty but rename of key for tag has data entered
      if (tag['key'].trim() == '' && tag['value'].trim().length > 0) {
        errors['auto_tag'] = 'Tag Names Cannot Be Empty';
        setHasErrors(true);
      }

      // only add tag if tag data exists in array
      if (tag['key'].trim().length > 0) {
        // key is tag you want to autotag, value is new name of tag's key
        tags[tag['key']] = {
          logic: 'Exists',
          key: tag['value'].trim() == '' ? null : tag['value'],
        };
      }
    });
  }
  requestOutputCollection.auto_tag = tags;

  // clear any old auto tags that have been removed
  if (initialOutputCollection && initialOutputCollection['auto_tag']) {
    Object.keys(initialOutputCollection.auto_tag).map((key) => {
      if (!requestOutputCollection.auto_tag[key]) {
        requestOutputCollection.auto_tag[key] = structuredClone(initialOutputCollection.auto_tag[key]);
        requestOutputCollection.auto_tag[key]['delete'] = true;
      }
    });
  }

  // clean up local structures used to map image info to renderable components
  delete requestOutputCollection.select_auto_tag;
  delete requestOutputCollection.select_groups;

  // check for errors, notify parent if they exist
  if (Object.keys(errors).length > 0) {
    setHasErrors(true);
  } else {
    setHasErrors(false);
  }
  setErrors(errors);
  setRequestOutputCollection(requestOutputCollection);
};

const OutputCollectionInputs = ({ initialOutputCollection, updateRequestOutputCollection, errors, disabled }) => {
  const [outputCollection, setOutputCollection] = useState(structuredClone(initialOutputCollection));
  // update output collection structure
  const updateOutputCollection = (key, subkey, value) => {
    // make a deep copy of the outputCollection
    const outputCollectionCopy = structuredClone(outputCollection);
    // set the new value for the key
    if (subkey) outputCollectionCopy[key][subkey] = value;
    else outputCollectionCopy[key] = value;
    // update the outputCollection object and trigger dom refresh
    setOutputCollection(outputCollectionCopy);
    updateRequestOutputCollection(outputCollectionCopy);
  };

  // this is needed for onload when cloning from an existing image
  useEffect(() => {
    updateRequestOutputCollection(initialOutputCollection);
    setOutputCollection(initialOutputCollection);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialOutputCollection]);

  return (
    <Row className="image-fields">
      <Form.Group className="mt-1">
        <Form.Label>
          <Subtitle>Results</Subtitle>
        </Form.Label>
        <OverlayTipRight tip={OutputCollectionToolTips.files.results}>
          <Form.Control
            type="text"
            value={outputCollection.files.results}
            placeholder="/tmp/thorium/results"
            disabled={disabled}
            onChange={(e) => updateOutputCollection('files', 'results', String(e.target.value).trim())}
          />
        </OverlayTipRight>
      </Form.Group>
      <Form.Group className="mt-1">
        <Form.Label>
          <Subtitle>Result Files</Subtitle>
        </Form.Label>
        <OverlayTipRight tip={OutputCollectionToolTips.files.result_files}>
          <Form.Control
            type="text"
            value={outputCollection.files.result_files}
            placeholder="/tmp/thorium/result-files"
            disabled={disabled}
            onChange={(e) => updateOutputCollection('files', 'result_files', String(e.target.value).trim())}
          />
        </OverlayTipRight>
      </Form.Group>
      <Form.Group className="mt-1">
        <Form.Label>
          <Subtitle>Result File Names</Subtitle>
        </Form.Label>
        <OverlayTipRight tip={OutputCollectionToolTips.files.names}>
          <SelectableArray
            initialEntries={outputCollection.files.names}
            setEntries={(fileNames) => updateOutputCollection('files', 'names', fileNames)}
            disabled={disabled}
            placeholder={'file name'}
          />
        </OverlayTipRight>
      </Form.Group>
      <Form.Group className="mt-1">
        <Form.Label>
          <Subtitle>Children</Subtitle>
        </Form.Label>
        <OverlayTipRight tip={OutputCollectionToolTips.children}>
          <Form.Control
            type="text"
            value={outputCollection.children}
            placeholder="/tmp/thorium/children"
            disabled={disabled}
            onChange={(e) => updateOutputCollection('children', '', String(e.target.value).trim())}
          />
        </OverlayTipRight>
      </Form.Group>
      <Form.Group className="mt-1">
        <Form.Label>
          <Subtitle>Tags</Subtitle>
        </Form.Label>
        <OverlayTipRight tip={OutputCollectionToolTips.files.tags}>
          <Form.Control
            type="text"
            value={outputCollection.files.tags}
            placeholder="/tmp/thorium/tags"
            disabled={disabled}
            onChange={(e) => updateOutputCollection('files', 'tags', String(e.target.value).trim())}
          />
        </OverlayTipRight>
      </Form.Group>
      <Row className="mt-2 mb-2">
        <Col>
          <Subtitle>Auto Tagging</Subtitle>
        </Col>
      </Row>
      <Row>
        <OverlayTipRight tip={OutputCollectionToolTips.auto_tag}>
          <SelectableDictionary
            entries={outputCollection.select_auto_tag}
            disabled={disabled}
            setEntries={(autoTags) => updateOutputCollection('select_auto_tag', '', autoTags)}
            keyPlaceholder={'key'}
            valuePlaceholder={'updated key (optional)'}
          />
        </OverlayTipRight>
        {errors && errors['auto_tag'] && (
          <center>
            <Alert variant="danger">{errors.auto_tag}</Alert>
          </center>
        )}
      </Row>
      <Row className="mt-2 mb-2">
        <Col>
          <Subtitle>Group Permissions</Subtitle>
        </Col>
      </Row>
      <Row>
        <OverlayTipRight tip={OutputCollectionToolTips.groups}>
          <center>
            <SelectGroups
              groups={outputCollection.select_groups}
              disabled={disabled}
              setGroups={(groups) => updateOutputCollection('select_groups', '', groups)}
            />
          </center>
        </OverlayTipRight>
      </Row>
    </Row>
  );
};

const EditOutputCollectionInputs = ({ initialOutputCollection, setUpdatedOutputCollection, errors, disabled }) => {
  return (
    <Row>
      <Col style={{ flex: 0.2 }}></Col>
      <Col style={{ flex: 1.25 }}></Col>
      <Col style={{ flex: 8 }}>
        <OutputCollectionInputs
          initialOutputCollection={initialOutputCollection}
          updateRequestOutputCollection={setUpdatedOutputCollection}
          errors={errors}
          disabled={disabled}
        />
      </Col>
    </Row>
  );
};

const CreateOutputCollectionFields = ({ initialOutputCollection, setUpdatedOutputCollection, errors, disabled }) => {
  return (
    <Fragment>
      <OutputCollectionInputs
        initialOutputCollection={initialOutputCollection}
        updateRequestOutputCollection={setUpdatedOutputCollection}
        errors={errors}
        disabled={disabled}
      />
    </Fragment>
  );
};

const ImageOutputCollection = ({ outputCollection, groups, setRequestOutputCollection, mode, setHasErrors, disabled }) => {
  const [errors, setErrors] = useState({});

  // provide the edit/create components with a callback to update a
  // request formatted outputCollection object
  const setUpdatedOutputCollection = (newOutputCollection) => {
    if (['Create', 'Copy'].includes(mode)) {
      return updateCreateRequestOutputCollection(newOutputCollection, setRequestOutputCollection, setErrors, setHasErrors);
    } else {
      return updateEditRequestOutputCollection(outputCollection, newOutputCollection, setRequestOutputCollection, setErrors, setHasErrors);
    }
  };

  if (outputCollection) {
    if (outputCollection) {
      // prebuild selected groups for buttons
      // start with all deselected and then apply existing selections if present
      const selectedGroups = {};
      if (groups) {
        groups.map((group) => {
          selectedGroups[group] = false;
        });
      }
      if (Object.keys(outputCollection).includes('groups') && outputCollection.groups.length > 0) {
        outputCollection.groups.map((group) => {
          selectedGroups[group] = true;
        });
      }
      // create new image
      if (mode == 'Create') {
        OutputCollectionTemplate['select_groups'] = selectedGroups;
        // copy/edit existing image
      } else {
        outputCollection['select_groups'] = selectedGroups;
      }
    }

    // prebuild auto_tagging for SelectableDictionary
    // It uses a [{'key': 'someKey', 'value': 'someValue}] structure
    // rather than {'key': {'logic': 'Exists', 'key': 'someRenamedKey'}} of auto_tag
    outputCollection['select_auto_tag'] = [];
    if (Object.keys(outputCollection).includes('auto_tag')) {
      const keys = Object.keys(outputCollection.auto_tag);
      const selectAutoTags = keys.map((key) => {
        const selectTag = {
          key: key,
          value: outputCollection.auto_tag[key]['key'],
        };
        return selectTag;
      });
      outputCollection['select_auto_tag'] = selectAutoTags;
    }
    // need an empty value here for new entries
    outputCollection['select_auto_tag'].push({ key: '', value: '' });
  }

  // copy mode requires cleanup of the duplicated image configuration before being placed into
  // the create component
  if (mode == 'Copy') {
    return (
      <Row>
        <Col className="title-col">
          <h5>Output Collection</h5>
        </Col>
        <Col className="field-col">
          <CreateOutputCollectionFields
            initialOutputCollection={outputCollection}
            setUpdatedOutputCollection={setUpdatedOutputCollection}
            errors={errors}
            disabled={disabled}
          />
        </Col>
      </Row>
    );
  } else if (mode == 'Create') {
    return (
      <Row>
        <Col className="title-col">
          <h5>Output Collection</h5>
        </Col>
        <Col className="field-col">
          <CreateOutputCollectionFields
            initialOutputCollection={OutputCollectionTemplate}
            setUpdatedOutputCollection={setUpdatedOutputCollection}
            errors={errors}
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
          <OverlayTipRight tip={OutputCollectionToolTips.self}>
            <b>{'Output Collection'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
      </Row>
      {mode == 'View' && <DisplayOutputCollection outputCollection={outputCollection} />}
      {mode == 'Edit' && (
        <EditOutputCollectionInputs
          initialOutputCollection={outputCollection}
          setUpdatedOutputCollection={setUpdatedOutputCollection}
          errors={errors}
          disabled={disabled}
        />
      )}
    </Fragment>
  );
};

export default ImageOutputCollection;
