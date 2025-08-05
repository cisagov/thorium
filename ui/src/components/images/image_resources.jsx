import React, { Fragment, useEffect, useState } from 'react';
import { Alert, Col, Form, Row } from 'react-bootstrap';
import { FaQuestionCircle } from 'react-icons/fa';

// project imports
import { FieldBadge, OverlayTipRight } from '@components';

const ResourcesToolTips = {
  self: `The resources required to run this image. Running images that exceed their requested 
    resources may be terminated.`,
  cpu: `The number of CPUs that an image will be allowed to consume. Requesting a large amount 
    of CPU may result in an image that can never be scheduled.  Units are either whole CPU or
    integer thousandths of a CPU (mCPU).`,
  memory: `The max amount of memory an image may be allowed to consume. Requesting a large 
    amount of memory may result in an image that can never be scheduled.`,
  ephemeral_storage: `The amount of ephemeral storage that this image requires to run. Requesting 
    a large amount of storage may result in an image that can never be scheduled.`,
  amd_gpu: `The number of AMD GPUs required to run this image. Requesting a large number of GPUs 
    may result in an image that can never be scheduled.`,
  nvidia_gpu: `The number of NVIDIA GPUs required to run this image. Requesting a large number of 
    GPUs may result in an image that can never be scheduled.`,
};

const ResourcesTemplate = {
  cpu: '',
  cpu_units: 'CPU',
  memory: '',
  memory_units: 'Gi',
  ephemeral_storage: '',
  ephemeral_storage_units: 'Gi',
  amd_gpu: '',
  nvidia_gpu: '',
};

const DisplayImageResources = ({ resources }) => {
  return (
    <Fragment>
      <Row>
        <Col>
          <OverlayTipRight tip={ResourcesToolTips.self}>
            <b>{'Resources'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
      </Row>
      {/* ************************** CPU ************************** */}
      <Row>
        <Col className="key-col-1" />
        <Col className="key-col-2-ext">
          <em>{`cpu: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ResourcesToolTips.cpu}>
              <FieldBadge field={`${String(parseInt(resources['cpu']))} mCPU`} color={'#7e7c7c'} />
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
      {/* ************************** Memory ************************** */}
      <Row>
        <Col className="key-col-1" />
        <Col className="key-col-2-ext">
          <em>{`memory: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ResourcesToolTips.memory}>
              <FieldBadge field={`${String(parseInt(resources['memory']))} MiB`} color={'#7e7c7c'} />
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
      {/* ************************** Ephemeral Storage ************************** */}
      <Row>
        <Col className="key-col-1" />
        <Col className="key-col-2-ext">
          <em>{`storage: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ResourcesToolTips.ephemeral_storage}>
              <FieldBadge field={`${resources['ephemeral_storage']} MiB`} color={'#7e7c7c'} />
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
      {/* ************************** Nvidia GPUs ************************** */}
      <Row>
        <Col className="key-col-1" />
        <Col className="key-col-2-ext">
          <em>{`nvidia gpu: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ResourcesToolTips.nvidia_gpu}>
              <FieldBadge field={resources['nvidia_gpu']} color={'#7e7c7c'} />
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
      {/* ************************** AMD GPUs ************************** */}
      <Row>
        <Col className="key-col-1" />
        <Col className="key-col-2-ext">
          <em>{`amd gpu: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ResourcesToolTips.amd_gpu}>
              <FieldBadge field={resources['amd_gpu']} color={'#7e7c7c'} />
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
    </Fragment>
  );
};

const updateCreateRequest = (resources, setRequestResources, setErrors, setHasErrors) => {
  const requestResources = structuredClone(resources);
  const errors = {};

  if (requestResources.worker_slots) {
    delete requestResources.worker_slots;
  }
  if (requestResources.cpu == '' && requestResources.memory) {
    errors['cpu'] = `cpu can't be empty when memory is specified`;
    delete requestResources.cpu;
  } else if (requestResources.cpu && requestResources.cpu_units == 'mCPU') {
    requestResources.cpu = String(`${requestResources.cpu}m`);
  } else if (requestResources.cpu) {
    requestResources.cpu = String(requestResources.cpu);
  }
  delete requestResources.cpu_units;

  if (requestResources.memory == '' && requestResources.cpu) {
    errors['memory'] = `memory can't be empty when cpu is specified`;
    delete requestResources.memory;
  } else if (requestResources.memory) {
    requestResources.memory = String(`${requestResources.memory}${requestResources.memory_units}`);
  }
  delete requestResources.memory_units;

  // if both CPU and memory are blank, don't submit in request
  if (requestResources.cpu == '' && requestResources.memory == '') {
    delete requestResources.memory;
    delete requestResources.cpu;
  }

  if (requestResources.ephemeral_storage == '') {
    delete requestResources.ephemeral_storage;
  } else {
    requestResources.ephemeral_storage = String(`${requestResources.ephemeral_storage}${requestResources.ephemeral_storage_units}`);
  }
  delete requestResources.ephemeral_storage_units;

  if (requestResources.amd_gpu == '') {
    delete requestResources.amd_gpu;
  } else {
    requestResources.amd_gpu = Number(requestResources.amd_gpu);
  }
  if (requestResources.nvidia_gpu == '') {
    delete requestResources.nvidia_gpu;
  } else {
    requestResources.nvidia_gpu = Number(requestResources.nvidia_gpu);
  }
  setErrors(errors);
  Object.keys(errors).length ? setHasErrors(true) : setHasErrors(false);
  setRequestResources(requestResources);
};

const updateEditRequest = (initialResources, resources, setRequestResources, setErrors, setHasErrors) => {
  const requestResources = structuredClone(resources);
  const errors = {};

  if (requestResources.worker_slots) {
    delete requestResources.worker_slots;
  }
  // both memory and cpu must be specified when one is set
  if (requestResources.cpu == '' && requestResources.memory != '') {
    errors['cpu'] = `cpu can't be empty when memory is specified`;
    delete requestResources.cpu;
  } else if (requestResources.cpu && requestResources.cpu_units == 'mCPU') {
    requestResources.cpu = String(`${requestResources.cpu}m`);
  } else if (requestResources.cpu) {
    requestResources.cpu = String(requestResources.cpu);
  }
  delete requestResources.cpu_units;

  // both memory and cpu must be specified when one is set
  if (requestResources.memory == '' && requestResources.cpu != '') {
    errors['memory'] = `memory can't be empty when cpu is specified`;
    delete requestResources.memory;
    // if both CPU and memory are blank, don't submit in request
  } else if (requestResources.memory) {
    requestResources.memory = String(`${requestResources.memory}${requestResources.memory_units}`);
  }
  delete requestResources.memory_units;

  // if both CPU and memory are blank, don't submit in request
  if (requestResources.cpu == '' && requestResources.memory == '') {
    delete requestResources.memory;
    delete requestResources.cpu;
  }

  if (requestResources.ephemeral_storage == '') {
    delete requestResources.ephemeral_storage;
  } else {
    requestResources.ephemeral_storage = String(`${requestResources.ephemeral_storage}${requestResources.ephemeral_storage_units}`);
  }
  delete requestResources.ephemeral_storage_units;

  if (requestResources.amd_gpu == '') {
    delete requestResources.amd_gpu;
  } else {
    requestResources.amd_gpu = Number(requestResources.amd_gpu);
  }
  if (requestResources.nvidia_gpu == '') {
    delete requestResources.nvidia_gpu;
  } else {
    requestResources.nvidia_gpu = Number(requestResources.nvidia_gpu);
  }
  if (Object.keys(requestResources).includes('worker_slots')) {
    delete requestResources['worker_slots'];
  }
  setErrors(errors);
  Object.keys(errors).length ? setHasErrors(true) : setHasErrors(false);
  setRequestResources(requestResources);
};

const ResourceFields = ({ initialResources, setRequestResources, errors }) => {
  const [resources, setResources] = useState(structuredClone(initialResources));

  // update a resource field based on the key and value
  const updateResources = (key, value) => {
    // make a deep copy of the dependency
    const resourcesCopy = structuredClone(resources);
    // set the new value for the key
    resourcesCopy[key] = value;
    // update the dependency object and trigger dom refreshsetRequestContext
    setResources(resourcesCopy);
    setRequestResources(resourcesCopy);
  };

  // this is needed for onload when cloning from an existing image
  useEffect(() => {
    setRequestResources(initialResources);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Fragment>
      {/* ************************** CPU ************************** */}
      <Row>
        <Col className="key-col-2-ext">
          <em>{`cpu: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ResourcesToolTips.cpu}>
              <Form.Group className="mb-2 image-fields">
                <Row>
                  <Col className="resource-type-col">
                    <Form.Control
                      type="text"
                      value={resources.cpu}
                      placeholder={resources.cpu_units == 'mCPU' ? '250' : '1'}
                      onChange={(e) => {
                        const validValue = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
                        updateResources('cpu', String(validValue));
                      }}
                    />
                  </Col>
                  <Col className="resource-unit-col">
                    <Form.Select value={resources.cpu_units} onChange={(e) => updateResources('cpu_units', String(e.target.value))}>
                      <option value="CPU">CPU</option>
                      <option value="mCPU">mCPU</option>
                    </Form.Select>
                  </Col>
                </Row>
                {errors && errors['cpu'] && (
                  <Alert variant="danger" className="d-flex justify-content-center m-2">
                    {errors.cpu}
                  </Alert>
                )}
              </Form.Group>
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
      {/* ************************** Memory ************************** */}
      <Row>
        <Col className="key-col-2-ext">
          <em>{`memory: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ResourcesToolTips.memory}>
              <Form.Group className="mb-2 image-fields">
                <Row>
                  <Col className="resource-type-col">
                    <Form.Control
                      type="text"
                      value={resources.memory}
                      placeholder={resources.memory_units == 'Mi' ? '250' : '1'}
                      onChange={(e) => {
                        const validValue = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
                        updateResources('memory', String(validValue));
                      }}
                    />
                  </Col>
                  <Col className="resource-unit-col">
                    <Form.Select value={resources.memory_units} onChange={(e) => updateResources('memory_units', String(e.target.value))}>
                      <option value="Gi">GiB</option>
                      <option value="Mi">MiB</option>
                    </Form.Select>
                  </Col>
                </Row>
                {errors && errors['memory'] && (
                  <Alert variant="danger" className="d-flex justify-content-center m-2">
                    {errors['memory']}
                  </Alert>
                )}
              </Form.Group>
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
      {/* ************************** Ephemeral Storage ************************** */}
      <Row>
        <Col className="key-col-2-ext">
          <em>{`storage: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ResourcesToolTips.ephemeral_storage}>
              <Form.Group className="mb-2 image-fields">
                <Row>
                  <Col className="resource-type-col">
                    <Form.Control
                      type="text"
                      value={resources.ephemeral_storage}
                      placeholder={resources.ephemeral_storage_units == 'Mi' ? '8192' : '8'}
                      onChange={(e) => {
                        const validValue = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
                        updateResources('ephemeral_storage', String(validValue));
                      }}
                    />
                  </Col>
                  <Col className="resource-unit-col">
                    <Form.Select
                      value={resources.ephemeral_storage_units}
                      onChange={(e) => updateResources('ephemeral_storage_units', String(e.target.value))}
                    >
                      <option value="Gi">GiB</option>
                      <option value="Mi">MiB</option>
                    </Form.Select>
                  </Col>
                </Row>
              </Form.Group>
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
      {/* ************************** Nvidia GPUs ************************** */}
      <Row>
        <Col className="key-col-2-ext">
          <em>{`nvidia gpu: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ResourcesToolTips.nvidia_gpu}>
              <Form.Group className="mb-2 image-fields">
                <Form.Control
                  type="text"
                  value={resources.nvidia_gpu ? resources.nvidia_gpu : ''}
                  placeholder="nvidia gpu"
                  onChange={(e) => {
                    const validValue = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
                    updateResources('nvidia_gpu', String(validValue));
                  }}
                />
              </Form.Group>
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
      {/* ************************** AMD GPUs ************************** */}
      <Row>
        <Col className="key-col-2-ext">
          <em>{`amd gpu: `}</em>
        </Col>
        <Col className="key-col-3">
          <div className="image-fields">
            <OverlayTipRight tip={ResourcesToolTips.amd_gpu}>
              <Form.Group className="mb-2 image-fields">
                <Form.Control
                  type="text"
                  value={resources.amd_gpu ? resources.amd_gpu : ''}
                  placeholder="amd gpu"
                  onChange={(e) => {
                    const validValue = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
                    updateResources('amd_gpu', String(validValue));
                  }}
                />
              </Form.Group>
            </OverlayTipRight>
          </div>
        </Col>
      </Row>
    </Fragment>
  );
};

const CreateImageResources = ({ resources, setRequestResources, errors }) => {
  return (
    <Row>
      <Col className="title-col">
        <h5>Resources</h5>
      </Col>
      <Col className="field-col">
        <ResourceFields initialResources={resources} errors={errors} setRequestResources={setRequestResources} />
      </Col>
    </Row>
  );
};

const EditImageResources = ({ resources, setRequestResources, errors }) => {
  return (
    <Row>
      {/* ************************** Resources ************************** */}
      <Row>
        <Col className="field-name-col-ext">
          <OverlayTipRight
            tip={`The resources required to run this image. Running images
                                that exceed their requested resources may be terminated.`}
          >
            <b>{'Resources'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
      </Row>
      <Row>
        <Col className="key-col-1" />
        <Col>
          <ResourceFields initialResources={resources} errors={errors} setRequestResources={setRequestResources} />
        </Col>
      </Row>
    </Row>
  );
};

const ImageResources = ({ resources, setRequestResources, setHasErrors, mode }) => {
  const [errors, setErrors] = useState({});
  // provide the edit/create components with a callback to update a
  // request formatted resources object
  const setUpdatedResources = (newResources) => {
    if (['Create', 'Copy'].includes(mode)) {
      return updateCreateRequest(newResources, setRequestResources, setErrors, setHasErrors);
    } else {
      return updateEditRequest(resources, newResources, setRequestResources, setErrors, setHasErrors);
    }
  };

  // units are always in `M` when coming from Thorium as in a copy or edit
  // when working with a fresh image, defaults will be `G`
  if (['Copy', 'Edit'].includes(mode)) {
    if (resources && !resources.cpu_units) {
      resources['cpu_units'] = 'mCPU';
    }
    if (resources && !resources.memory_units) {
      resources['memory_units'] = 'Mi';
    }
    if (resources && !resources.ephemeral_storage_units) {
      resources['ephemeral_storage_units'] = 'Mi';
    }
  }

  if (mode == 'Copy') {
    return <CreateImageResources resources={resources} errors={errors} setRequestResources={setUpdatedResources} />;
  } else if (mode == 'Create') {
    return <CreateImageResources resources={ResourcesTemplate} errors={errors} setRequestResources={setUpdatedResources} />;
  }

  return (
    <Fragment>
      {mode == 'View' && <DisplayImageResources resources={resources} />}
      {mode == 'Edit' && <EditImageResources resources={resources} errors={errors} setRequestResources={setUpdatedResources} />}
    </Fragment>
  );
};

export default ImageResources;
