import React, { Fragment, useEffect, useState } from 'react';
import { Col, Form, Row } from 'react-bootstrap';
import { FaQuestionCircle } from 'react-icons/fa';

// project imports
import { FieldBadge, OverlayTipRight, SelectableArray, SelectableDictionary, Subtitle } from '@components';

const DepToolTips = {
  self: `The dependencies an image needs to run. These might
          include files, repos, results from other tools, or
          ephemeral files that passed in as reaction arguments.`,
  samples: {
    location: `Destination path to download file(s) into when running this image.`,
    kwarg: `Argument used to pass this image the path to sample dependencies: --some-arg.`,
    strategy: `The method used to pass in dependencies as arguments: file path, file names, 
                directory path, or disabled.`,
  },
  repos: {
    location: `Destination path to download repo(s) into when running this image.`,
    kwarg: `Argument for passing this image the path to dependencies: --some-arg.`,
    strategy: `The method used to pass in dependencies as arguments: file path, 
                directory path, or disabled.`,
  },
  results: {
    location: `Destination path to download result(s) from other tools into when running 
                this image.`,
    kwarg: `Arguments used to pass the result dependency paths to the image.`,
    strategy: `The method used to pass in dependencies as arguments: file path, directory path, 
                or disabled.`,
    names: `A list of result file names that this image depends on from the dependent images. 
              Default behavior is to pull all results from the images.`,
    images: `A list of Thorium images to pull results from for this image. You must select a 
              group for the image you are creating to see the available images for the result
              dependency.`,
  },
  ephemeral: {
    location: `Destination path to download ephemeral files into when running this image.`,
    kwarg: `Argument for passing this image the path to ephemeral file dependencies: --some-arg.`,
    strategy: `The method used to pass in dependencies as arguments: file path, directory path, 
                or disabled.`,
    names: `A list of ephemeral file names that this image depends on. Ephemeral files are passed 
              into the reaction and are purged after a reaction is completed.`,
  },
  tags: {
    enabled: `Whether tags for target samples will be downloaded before this image is run.`,
    kwarg: `Argument for passing this image the path to the JSON formatted tags file: 
      --some-arg.`,
    location: `Destination path to download JSON tags file to when running this image.`,
    strategy: `The method used to pass in tags dependencies as arguments: file path, directory 
      path, or disabled.`,
  },
};

const DependencyTemplate = {
  samples: {
    location: '', // string
    kwarg: '', // string or map (object)
    strategy: 'Paths', // DependencyStrategy enums
  },
  repos: {
    location: '', // string
    kwarg: '', // string or map (object)
    strategy: 'Paths', // DependencyStrategy enums
  },
  results: {
    location: '', // string
    kwarg: { List: '' }, // string or map (object)
    kwargList: 'List', // List of options for kwargs
    strategy: 'Paths', // DependencyStrategy enums
    names: [''], // array of strings
    images: [''], // array of strings
  },
  ephemeral: {
    location: '', // string
    kwarg: '', // string or map (object)
    strategy: 'Paths', // DependencyStrategy enums
    names: [''], // array of strings
  },
  tags: {
    location: '', // string
    kwarg: '', // string (optional)
    strategy: 'Paths', // DependencyStrategy enums
    enabled: false, // boolean
  },
};
// The Thorium enum for dependency strategies
const DependencyStrategies = ['Paths', 'Names', 'Directory', 'Disabled'];

// The Thorium enum for results kwargs
const ResultsKwarg = ['List', 'Map', 'None'];

const DisplayDependencyFields = ({ dependencies }) => {
  return (
    <Fragment>
      {dependencies && Object.keys(dependencies).length > 0 && (
        <Fragment>
          <Row>
            {/* --------- Samples ---------*/}
            <Col style={{ flex: 0.1 }}></Col>
            <Col style={{ flex: 1 }}>
              <em>{`samples: `}</em>
            </Col>
            <Col style={{ flex: 8 }}></Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`location: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['samples']['location']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`kwarg: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['samples']['kwarg']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`strategy: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['samples']['strategy']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
          {/* --------- Repos ---------*/}
          <Row>
            <Col style={{ flex: 0.1 }}></Col>
            <Col style={{ flex: 1 }}>
              <em>{`repos: `}</em>
            </Col>
            <Col style={{ flex: 8 }}></Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`location: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['repos']['location']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`kwarg: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['repos']['kwarg']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`strategy: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['repos']['strategy']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
          {/* --------- Results ---------*/}
          <Row>
            <Col style={{ flex: 0.1 }}></Col>
            <Col style={{ flex: 1 }}>
              <em>{`results: `}</em>
            </Col>
            <Col style={{ flex: 8 }}></Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`images: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['results']['images']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
          {dependencies['results']['images'].length > 0 && (
            <>
              <Row>
                <Col style={{ flex: 1 }}></Col>
                <Col style={{ flex: 3 }}>
                  <em>{`location: `}</em>
                </Col>
                <Col style={{ flex: 21.5 }}>
                  <Row>
                    <Col>
                      <FieldBadge field={dependencies['results']['location']} color={'#7e7c7c'} />
                    </Col>
                  </Row>
                </Col>
              </Row>
              <Row>
                <Col style={{ flex: 1 }}></Col>
                <Col style={{ flex: 3 }}>
                  <em>{`kwarg: `}</em>
                </Col>
                <Col style={{ flex: 21.5 }}>
                  <Row>
                    <Col>
                      {dependencies['results']['kwarg']['Map'] && (
                        <FieldBadge field={dependencies['results']['kwarg']['Map']} color={'#7e7c7c'} />
                      )}
                      {!dependencies['results']['kwarg']['Map'] && (
                        <FieldBadge field={dependencies['results']['kwarg']} color={'#7e7c7c'} />
                      )}
                    </Col>
                  </Row>
                </Col>
              </Row>
              <Row>
                <Col style={{ flex: 1 }}></Col>
                <Col style={{ flex: 3 }}>
                  <em>{`strategy: `}</em>
                </Col>
                <Col style={{ flex: 21.5 }}>
                  <Row>
                    <Col>
                      <FieldBadge field={dependencies['results']['strategy']} color={'#7e7c7c'} />
                    </Col>
                  </Row>
                </Col>
              </Row>
              <Row>
                <Col style={{ flex: 1 }}></Col>
                <Col style={{ flex: 3 }}>
                  <em>{`names: `}</em>
                </Col>
                <Col style={{ flex: 21.5 }}>
                  <Row>
                    <Col>
                      <FieldBadge field={dependencies['results']['names']} color={'#7e7c7c'} />
                    </Col>
                  </Row>
                </Col>
              </Row>
            </>
          )}
          {/* --------- Ephemeral ---------*/}
          <Row>
            <Col style={{ flex: 0.1 }}></Col>
            <Col style={{ flex: 1 }}>
              <em>{`ephemeral: `}</em>
            </Col>
            <Col style={{ flex: 8 }}></Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`location: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['ephemeral']['location']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`kwarg: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['ephemeral']['kwarg']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`strategy: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['ephemeral']['strategy']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`names: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['ephemeral']['names']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
          {/* --------- Tags ---------*/}
          <Row>
            <Col style={{ flex: 0.1 }}></Col>
            <Col style={{ flex: 1 }}>
              <em>{`tags: `}</em>
            </Col>
            <Col style={{ flex: 8 }}></Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`enabled: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['tags']['enabled']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`location: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['tags']['location']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`kwarg: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['tags']['kwarg']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
          <Row>
            <Col style={{ flex: 1 }}></Col>
            <Col style={{ flex: 3 }}>
              <em>{`strategy: `}</em>
            </Col>
            <Col style={{ flex: 21.5 }}>
              <Row>
                <Col>
                  <FieldBadge field={dependencies['tags']['strategy']} color={'#7e7c7c'} />
                </Col>
              </Row>
            </Col>
          </Row>
        </Fragment>
      )}
    </Fragment>
  );
};

// change form structure to requests structure for creating an image
const updateCreateRequestDependencies = (newDependencies, setRequestDependencies) => {
  const requestDependencies = structuredClone(newDependencies);

  // SAMPLES
  if (requestDependencies.samples.location == '') {
    delete requestDependencies.samples.location;
  }
  if (requestDependencies.samples.kwarg == '' || requestDependencies.samples.kwarg == null) {
    delete requestDependencies.samples.kwarg;
  }
  // REPOS
  if (requestDependencies.repos.location == '') {
    delete requestDependencies.repos.location;
  }
  if (requestDependencies.repos.kwarg == '' || requestDependencies.repos.kwarg == null) {
    delete requestDependencies.repos.kwarg;
  }
  // RESULTS
  if (requestDependencies.results.location == '') {
    delete requestDependencies.results.location;
  }

  if (requestDependencies.results.kwarg['List'] == '' || requestDependencies.results.kwarg == null) {
    delete requestDependencies.results.kwarg;
  }
  if (requestDependencies.results.names.length == 1 && requestDependencies.results.names[0] == '') {
    delete requestDependencies.results.names;
  } else {
    // clear out blank file names
    requestDependencies.results.names = requestDependencies.results.names.filter((name) => name != '');
  }
  if (requestDependencies.results.images.length == 1 && requestDependencies.results.images[0] == '') {
    delete requestDependencies.results.images;
  } else {
    // clear out blank image names
    requestDependencies.results.images = requestDependencies.results.images.filter((imageName) => imageName != '');
  }
  if (requestDependencies.results['kwargList']) {
    delete requestDependencies.results['kwargList'];
  }
  // EPHEMERAL
  if (requestDependencies.ephemeral.location == '') {
    delete requestDependencies.ephemeral.location;
  }
  if (requestDependencies.ephemeral.kwarg == '' || requestDependencies.ephemeral.kwarg == null) {
    delete requestDependencies.ephemeral.kwarg;
  }
  if (requestDependencies.ephemeral.names.length == 1 && requestDependencies.ephemeral.names[0] == '') {
    delete requestDependencies.ephemeral.names;
  } else {
    // clear out blank file names
    requestDependencies.ephemeral.names = requestDependencies.ephemeral.names.filter((name) => name != '');
  }

  // TAGS
  if (!requestDependencies.tags.enabled) {
    delete requestDependencies.tags.location;
    delete requestDependencies.tags.kwarg;
    delete requestDependencies.tags.strategy;
  }
  // don't pass in location if not specified (use Thorium default)
  if (requestDependencies.tags.enabled && requestDependencies.tags.location == '') {
    delete requestDependencies.tags.location;
  }

  // cleanup empty keys to prevent 422s
  Object.keys(requestDependencies).map((key) => {
    if (Object.values(requestDependencies[key]).length == 0) {
      delete requestDependencies[key];
    }
  });

  // set the built request format for updating dependencies
  setRequestDependencies(requestDependencies);
};

// change form structure to requests structure for editing an image
const updateEditRequestDependencies = (initialDependencies, newDependencies, setRequestDependencies) => {
  // start with the updated dependencies for the request and
  // then modify it for the request format
  const requestDependencies = structuredClone(newDependencies);

  // SAMPLES
  if (requestDependencies.samples.kwarg == '') {
    delete requestDependencies.samples.kwarg;
    requestDependencies.samples['clear_kwarg'] = true;
  }
  // REPOS
  if (requestDependencies.repos.kwarg == '') {
    delete requestDependencies.repos.kwarg;
    requestDependencies.repos['clear_kwarg'] = true;
  }
  // RESULTS
  if (
    requestDependencies.results.kwarg == '' ||
    requestDependencies.results.kwarg['List'] == '' ||
    requestDependencies.results.kwarg == null
  ) {
    delete requestDependencies.results.kwarg;
    requestDependencies.results['clear_kwarg'] = true;
  }

  // clear out blank file names
  requestDependencies.results.names = requestDependencies.results.names.filter((name) => name != '');
  // filter out all the names that were removed
  if (initialDependencies.results.names.length > 0) {
    requestDependencies.results['remove_names'] = initialDependencies.results.names.filter(
      (value) => !requestDependencies.results.names.includes(value),
    );
  }
  // filter out all names that were added, these must be new names as we ignore duplicates
  if (requestDependencies.results.names.length > 0) {
    requestDependencies.results['add_names'] = requestDependencies.results.names.filter(
      (value) => !initialDependencies.results.names.includes(value),
    );
  }
  // names isn't needed for updates, only add_names and remove_names
  delete requestDependencies.results.names;

  // filter out all the images that were removed
  if (initialDependencies.results.names.length > 0) {
    requestDependencies.results['remove_images'] = initialDependencies.results.images.filter(
      (value) => !requestDependencies.results.images.includes(value),
    );
  }
  // filter out all images that were added, these must be new images as we ignore duplicates
  if (requestDependencies.results.images.length > 0) {
    requestDependencies.results['add_images'] = requestDependencies.results.images.filter(
      (value) => !initialDependencies.results.images.includes(value),
    );
  }
  // images isn't needed for updates, only add_images and remove_images
  delete requestDependencies.results.images;

  // EPHEMERAL
  if (requestDependencies.ephemeral.kwarg == '') {
    delete requestDependencies.ephemeral.kwarg;
    requestDependencies.ephemeral['clear_kwarg'] = true;
  }
  // clear out blank file names
  requestDependencies.ephemeral.names = requestDependencies.ephemeral.names.filter((name) => name != '');
  // filter out all the names that were removed
  if (initialDependencies.ephemeral.names.length > 0) {
    requestDependencies.ephemeral['remove_names'] = initialDependencies.ephemeral.names.filter(
      (value) => !requestDependencies.ephemeral.names.includes(value),
    );
  }
  // filter out all names that were added, these must be new names as we ignore duplicates
  if (requestDependencies.ephemeral.names.length > 0) {
    requestDependencies.ephemeral['add_names'] = requestDependencies.ephemeral.names.filter(
      (value) => !initialDependencies.ephemeral.names.includes(value),
    );
  }
  // names isn't needed for updates, only add_names and remove_names
  delete requestDependencies.ephemeral.names;

  // set the built request format for updating dependencies
  setRequestDependencies(requestDependencies);
};

const DependencyInputs = ({ images, initialDependencies, updateRequestDependencies, disabled }) => {
  const [dependencies, setDependencies] = useState(structuredClone(initialDependencies));
  // array for results.kwarg.map
  const [resultsKwargsMap, setResultsKwargsMap] = useState(
    initialDependencies &&
      initialDependencies.results.kwarg &&
      initialDependencies.results.kwarg.Map &&
      Object.keys(initialDependencies.results.kwarg.Map).length
      ? Object.keys(initialDependencies.results.kwarg.Map).map((item) => {
          return {
            key: item,
            value: initialDependencies.results.kwarg.Map[item],
          };
        })
      : [{ key: '', value: '' }],
  );

  // update a <dependency>'s <key> with new <value>
  const updateDependency = (dependency, key, value) => {
    // make a deep copy of the dependency
    const dependencyCopy = structuredClone(dependencies);
    // if kwargs is set to map convert value into thorium format
    if (key == 'kwarg' && typeof value == 'object' && Object.keys(value).includes('Map')) {
      const validKwargMap = {};
      value['Map'].map((variable) => {
        // a valid key must be set
        if (variable['key']) {
          validKwargMap[variable['key']] = variable['value'];
        }
      });
      value = { Map: validKwargMap };
    }
    // if kwargList option None is selected
    if (key == 'kwargList' && value == 'None') {
      dependencyCopy[dependency]['kwarg'] = value;
    }

    // set the new value for the key
    dependencyCopy[dependency][key] = value;
    // update the dependency object and trigger dom refresh
    setDependencies(dependencyCopy);
    updateRequestDependencies(dependencyCopy);
  };

  // this is needed for onload when cloning from an existing image
  useEffect(() => {
    updateRequestDependencies(initialDependencies);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Fragment>
      <Row className="image-fields">
        <Col className="output-col">
          <h6>Samples</h6>
        </Col>
      </Row>
      <Row className="image-fields">
        <Form.Group>
          <Form.Label>
            <Subtitle>Location</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.samples.location}>
            <Form.Control
              type="text"
              value={dependencies.samples.location}
              placeholder="/tmp/thorium/samples"
              disabled={disabled}
              onChange={(e) => updateDependency('samples', 'location', String(e.target.value).trim())}
            />
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>kwarg</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.samples.kwarg}>
            <Form.Control
              type="text"
              value={dependencies.samples.kwarg ? dependencies.samples.kwarg : ''}
              placeholder="--input-file"
              disabled={disabled}
              onChange={(e) => updateDependency('samples', 'kwarg', String(e.target.value).trim())}
            />
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>Strategy</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.samples.strategy}>
            <Form.Select
              value={dependencies.samples.strategy}
              disabled={disabled}
              onChange={(e) => updateDependency('samples', 'strategy', String(e.target.value))}
            >
              {DependencyStrategies.map((strategy) => (
                <option key={strategy} value={strategy}>
                  {strategy}
                </option>
              ))}
            </Form.Select>
          </OverlayTipRight>
        </Form.Group>
      </Row>
      <hr />
      <Row className="image-fields">
        <Col className="output-col">
          <h6>Repos</h6>
        </Col>
      </Row>
      <Row className="image-fields">
        <Form.Group>
          <Form.Label>
            <Subtitle>Location</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.repos.location}>
            <Form.Control
              type="text"
              value={dependencies.repos.location}
              placeholder="/tmp/thorium/repos"
              disabled={disabled}
              onChange={(e) => updateDependency('repos', 'location', String(e.target.value).trim())}
            />
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>kwarg</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.repos.kwarg}>
            <Form.Control
              type="text"
              value={dependencies.repos.kwarg ? dependencies.repos.kwarg : ''}
              placeholder="--input-repo"
              disabled={disabled}
              onChange={(e) => updateDependency('repos', 'kwarg', String(e.target.value).trim())}
            />
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>Strategy</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.repos.strategy}>
            <Form.Select
              value={dependencies.repos.strategy}
              disabled={disabled}
              onChange={(e) => updateDependency('repos', 'strategy', String(e.target.value))}
            >
              {DependencyStrategies.map((strategy) => (
                <option key={strategy} value={strategy}>
                  {strategy}
                </option>
              ))}
            </Form.Select>
          </OverlayTipRight>
        </Form.Group>
      </Row>
      <hr />
      <Row className="image-fields">
        <Col className="output-col">
          <h6>Results</h6>
        </Col>
      </Row>
      <Row className="image-fields">
        <Form.Group>
          <Form.Label>
            <Subtitle>Location</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.results.location}>
            <Form.Control
              type="text"
              value={dependencies.results.location}
              placeholder="/tmp/thorium/prior-results"
              disabled={disabled}
              onChange={(e) => updateDependency('results', 'location', String(e.target.value).trim())}
            />
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>kwarg</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.results.kwarg}>
            <Form.Select
              className="mb-3"
              value={dependencies.results.kwargList}
              disabled={disabled}
              onChange={(e) => updateDependency('results', 'kwargList', String(e.target.value))}
            >
              {ResultsKwarg.map((kwargOption) => (
                <option key={kwargOption} value={kwargOption}>
                  {kwargOption}
                </option>
              ))}
            </Form.Select>
            {dependencies.results.kwargList == 'Map' && (
              <SelectableDictionary
                entries={resultsKwargsMap}
                disabled={false}
                setEntries={(updatedMap) => {
                  setResultsKwargsMap(updatedMap);
                  updateDependency('results', 'kwarg', { Map: updatedMap });
                }}
                keyPlaceholder={'New Variable'}
                valuePlaceholder={'New Value'}
                trim={true}
                keys={images}
              />
            )}
            {dependencies.results.kwargList == 'List' && (
              <Form.Control
                type="text"
                value={dependencies.results.kwarg['List'] ? dependencies.results.kwarg['List'] : ''}
                placeholder="--input-result"
                disabled={disabled}
                onChange={(e) =>
                  updateDependency('results', 'kwarg', {
                    List: String(e.target.value).trim(),
                  })
                }
              />
            )}
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>Strategy</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.results.strategy}>
            <Form.Select
              value={dependencies.results.strategy}
              disabled={disabled}
              onChange={(e) => updateDependency('results', 'strategy', String(e.target.value))}
            >
              {DependencyStrategies.map((strategy) => (
                <option key={strategy} value={strategy}>
                  {strategy}
                </option>
              ))}
            </Form.Select>
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>File Names</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.results.names}>
            <SelectableArray
              initialEntries={dependencies.results.names}
              setEntries={(fileNames) => updateDependency('results', 'names', fileNames)}
              disabled={disabled}
              placeholder={'Filename'}
            />
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>Images</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.results.images}>
            <SelectableArray
              initialEntries={dependencies.results.images}
              setEntries={(imageNames) => updateDependency('results', 'images', imageNames)}
              disabled={disabled || images.length == 0}
              placeholder={images}
            />
          </OverlayTipRight>
        </Form.Group>
      </Row>
      <hr />
      <Row className="image-fields">
        <Col className="output-col">
          <h6>Ephemeral</h6>
        </Col>
      </Row>
      <Row className="image-fields">
        <Form.Group>
          <Form.Label>
            <Subtitle>Location</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.ephemeral.location}>
            <Form.Control
              type="text"
              value={dependencies.ephemeral.location}
              placeholder="/tmp/thorium/ephemeral"
              disabled={disabled}
              onChange={(e) => updateDependency('ephemeral', 'location', String(e.target.value).trim())}
            />
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>kwarg</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.ephemeral.kwarg}>
            <Form.Control
              type="text"
              value={dependencies.ephemeral.kwarg ? dependencies.ephemeral.kwarg : ''}
              placeholder="--ephemeral-file"
              disabled={disabled}
              onChange={(e) => updateDependency('ephemeral', 'kwarg', String(e.target.value).trim())}
            />
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>Strategy</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.ephemeral.strategy}>
            <Form.Select
              value={dependencies.ephemeral.strategy}
              disabled={disabled}
              onChange={(e) => updateDependency('ephemeral', 'strategy', String(e.target.value))}
            >
              {DependencyStrategies.map((strategy) => (
                <option key={strategy} value={strategy}>
                  {strategy}
                </option>
              ))}
            </Form.Select>
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>File Names</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.ephemeral.names}>
            <SelectableArray
              initialEntries={dependencies.ephemeral.names}
              setEntries={(fileNames) => updateDependency('ephemeral', 'names', fileNames)}
              disabled={disabled}
              placeholder={'Filename'}
            />
          </OverlayTipRight>
        </Form.Group>
      </Row>
      <hr />
      <Row className="image-fields">
        <Col className="output-col">
          <h6>Tags</h6>
        </Col>
      </Row>
      <Row className="image-fields">
        <Form.Group className="d-flex inline">
          <Form.Label className="mt-1">
            <Subtitle>Enabled</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.tags.enabled}>
            <h6>
              <Form.Check
                className="ms-3"
                type="switch"
                id="is-generator"
                label=""
                checked={dependencies.tags.enabled}
                disabled={disabled}
                onChange={(e) => updateDependency('tags', 'enabled', !dependencies.tags.enabled)}
              />
            </h6>
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>Location</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.tags.location}>
            <Form.Control
              type="text"
              value={dependencies.tags.location}
              placeholder="/tmp/thorium/prior-tags"
              disabled={disabled}
              onChange={(e) => updateDependency('tags', 'location', String(e.target.value).trim())}
            />
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>kwarg</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.tags.kwarg}>
            <Form.Control
              type="text"
              value={dependencies.tags.kwarg ? dependencies.tags.kwarg : ''}
              placeholder="--prior-tags"
              disabled={disabled}
              onChange={(e) => updateDependency('tags', 'kwarg', String(e.target.value).trim())}
            />
          </OverlayTipRight>
        </Form.Group>
        <Form.Group>
          <Form.Label>
            <Subtitle>Strategy</Subtitle>
          </Form.Label>
          <OverlayTipRight tip={DepToolTips.tags.strategy}>
            <Form.Select
              value={dependencies.tags.strategy}
              disabled={disabled}
              onChange={(e) => updateDependency('tags', 'strategy', String(e.target.value))}
            >
              {DependencyStrategies.map((strategy) => (
                <option key={strategy} value={strategy}>
                  {strategy}
                </option>
              ))}
            </Form.Select>
          </OverlayTipRight>
        </Form.Group>
      </Row>
    </Fragment>
  );
};

const EditDependencyFields = ({ initialDependencies, images, setUpdatedDependencies, disabled }) => {
  const dependencies = structuredClone(initialDependencies);

  // these fields aren't always filled in, but need to be
  if (!dependencies.ephemeral['kwarg']) {
    dependencies.ephemeral['kwarg'] = '';
  }
  if (!dependencies.repos['kwarg']) {
    dependencies.repos['kwarg'] = '';
  }
  if (!dependencies.results['kwarg'] || dependencies.results['kwarg'] == 'None') {
    dependencies.results['kwarg'] = '';
  }
  if (!dependencies.samples['kwarg']) {
    dependencies.samples['kwarg'] = '';
  }
  // add selectable kwargList
  if (!dependencies.results['kwargList']) {
    dependencies.results['kwargList'] = Object.keys(dependencies.results['kwarg']).length
      ? Object.keys(dependencies.results['kwarg'])[0]
      : 'None';
  }
  return (
    <Row>
      <Col style={{ flex: 0.2 }}></Col>
      <Col style={{ flex: 1.25 }}></Col>
      <Col style={{ flex: 8 }}>
        <DependencyInputs
          images={images}
          initialDependencies={dependencies}
          updateRequestDependencies={setUpdatedDependencies}
          disabled={disabled}
        />
      </Col>
    </Row>
  );
};

const CreateDependencyFields = ({ images, mode, initialDependencies, setCreateDependencies, disabled }) => {
  if (mode && mode == 'Copy') {
    // None, List, Map
    const kwargOption = initialDependencies.results['kwarg'];
    // add selectable kwargList
    if (!initialDependencies.results['kwargList']) {
      initialDependencies.results['kwargList'] = kwargOption == 'None' ? 'None' : Object.keys(initialDependencies.results['kwarg'])[0];
    }
  }

  return (
    <Fragment>
      <DependencyInputs
        images={images}
        initialDependencies={initialDependencies}
        updateRequestDependencies={setCreateDependencies}
        disabled={disabled}
      />
    </Fragment>
  );
};

// Component to display dependency fields in image creation, editing or viewing existing images
const ImageDependencies = ({ dependencies, images, setRequestDependencies, mode, disabled }) => {
  // ensure arrays in dependencies have values, they must be blank strings
  if (
    dependencies &&
    'results' in dependencies &&
    'images' in dependencies.results &&
    Array.isArray(dependencies.results.images) &&
    dependencies.results.images.length == 0
  ) {
    dependencies.results.images = [''];
  }
  if (
    dependencies &&
    'results' in dependencies &&
    'names' in dependencies.results &&
    Array.isArray(dependencies.results.names) &&
    dependencies.results.names.length == 0
  ) {
    dependencies.results.names = [''];
  }
  if (
    dependencies &&
    'ephemeral' in dependencies &&
    'names' in dependencies.ephemeral &&
    Array.isArray(dependencies.ephemeral.names) &&
    dependencies.ephemeral.names.length == 0
  ) {
    dependencies.ephemeral.names = [''];
  }

  // provide the edit/create components with a callback to update a
  // request formatted dependencies object
  const setUpdatedDependences = (newDependencies) => {
    if (['Create', 'Copy'].includes(mode)) {
      return updateCreateRequestDependencies(newDependencies, setRequestDependencies);
    } else {
      return updateEditRequestDependencies(dependencies, newDependencies, setRequestDependencies);
    }
  };

  // copy mode requires cleanup of the duplicated image configuration before being placed into
  // the create component
  if (mode == 'Copy') {
    return (
      <Row>
        <Col className="title-col">
          <h5>Dependencies</h5>
        </Col>
        <Col className="field-col">
          <CreateDependencyFields
            initialDependencies={dependencies}
            mode={'Copy'}
            images={images}
            setCreateDependencies={setUpdatedDependences}
            disabled={disabled}
          />
        </Col>
      </Row>
    );
  } else if (mode == 'Create') {
    return (
      <Row>
        <Col className="title-col">
          <h5>Dependencies</h5>
        </Col>
        <Col className="field-col">
          <CreateDependencyFields
            initialDependencies={DependencyTemplate}
            images={images}
            setCreateDependencies={setUpdatedDependences}
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
          <OverlayTipRight tip={DepToolTips.self}>
            <b>{'Dependencies'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
      </Row>
      {mode == 'View' && <DisplayDependencyFields dependencies={dependencies} />}
      {mode == 'Edit' && (
        <EditDependencyFields
          initialDependencies={dependencies}
          images={images}
          setUpdatedDependencies={setUpdatedDependences}
          disabled={disabled}
        />
      )}
    </Fragment>
  );
};

export default ImageDependencies;
