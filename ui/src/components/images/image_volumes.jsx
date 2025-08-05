import React, { Fragment, useEffect, useState } from 'react';
import { Alert, Button, Col, Form, Row } from 'react-bootstrap';
import { FaTrash } from 'react-icons/fa';
import { FaQuestionCircle } from 'react-icons/fa';

// project imports
import { FieldBadge, OverlayTipRight, Subtitle } from '@components';

const VolumesToolTips = {
  self: `Kubernetes volumes to map in during image run. Volumes can be host paths or 
    configuration files.`,
  name: `The name of this volume. Names must be unique and can only contain alphanumeric 
    characters and dashes.`,
  archetype: `The volume type. Volume types determine mount behavior and the volume source.`,
  mode: `The linux file permissions for mounted volumes (octal).`,
  optional: `Whether this volume is required to run the image.`,
  host_path: `Path on host to mount into the image.`,
  host_path_type: `The type of the source target being mounted.`,
  nfs_server: `The NFS server hostname to mount the NFS share from.`,
  nfs_path: `The NFS share path to mount from the server.`,
  mount_path: `The volume's mount path within the running container image.`,
  sub_path: `The file or directory from the volume to mount.`,
  read_only: `Whether the content in this volume can be modified at runtime. Read only is 
    recommended for most tools.`,
  kustomize: `Whether or not this volume was created manually or through kustomize. Kustomize 
    volumes can only be created by admins.`,
};

const NewVolumeTemplate = {
  name: '', // required always
  archetype: '', // required always
  mount_path: '', // required always
  sub_path: '', // optional, default none
  read_only: false, // optional, default false
  kustomize: false, // optional, default false
  config_map: {
    default_mode: '', // optional
    optional: false, // optional
  },
  secret: {
    default_mode: '', // optional
    optional: false, // optional
  },
  nfs: {
    path: '', // required for NFS archetype
    server: '', // required for NFS archetype
  },
  host_path: {
    path: '', // required for HostPath archetype
    path_type: '', // optional, default none
  },
  errors: {
    name: 'Required',
    archetype: 'Required',
    mount_path: 'Required',
  },
};

// Possible values of archetype for image creation request and display format
// Note: these differ from the thorium formated keys:
//    const Archetypes = ['config_map', 'secret', 'host_path', 'nfs'];
const DisplayArchetypes = ['ConfigMap', 'Secret', 'HostPath', 'NFS'];

// Possible values for the PathType enum value of the host_path Archetype
const HostPathTypes = ['DirectoryOrCreate', 'Directory', 'FileOrCreate', 'File', 'Socket', 'CharDevice', 'BlockDevice'];

const DisplayVolumes = ({ volumes }) => {
  return (
    <Fragment>
      {volumes &&
        volumes.map((volume, idx) => (
          <Row key={idx}>
            <Col>
              {/* FIELDS: iterate through each field per volume not in edit mode */}
              {Object.entries(volume)
                .filter(([key, value]) => key != 'errors')
                .map(([key, value], idx) => (
                  <Row key={key}>
                    <Col style={{ flex: 0.2 }}></Col>
                    <Col style={{ flex: 1.25 }}>
                      <em>{`${key}: `}</em>
                    </Col>
                    <Col style={{ flex: 8 }}>
                      <FieldBadge field={JSON.stringify(value)} color={'#7e7c7c'} />
                    </Col>
                  </Row>
                ))}
              {volumes.length != idx + 1 ? <hr /> : null}
            </Col>
          </Row>
        ))}
    </Fragment>
  );
};

const filterVolumeFields = (volumes) => {
  const filteredVolumes = structuredClone(volumes);
  // need to delete unused fields from each volume template
  // this is really just a result of "populating the template" with all
  // possible values for all Archetypes which makes working with useState
  // functions of the forms easier
  filteredVolumes.forEach(function (volume, idx) {
    switch (volume['archetype']) {
      case 'ConfigMap':
        delete filteredVolumes[idx]['secret'];
        delete filteredVolumes[idx]['host_path'];
        delete filteredVolumes[idx]['nfs'];
        break;
      case 'Secret':
        delete filteredVolumes[idx]['config_map'];
        delete filteredVolumes[idx]['host_path'];
        delete filteredVolumes[idx]['nfs'];
        break;
      case 'HostPath':
        delete filteredVolumes[idx]['config_map'];
        delete filteredVolumes[idx]['secret'];
        delete filteredVolumes[idx]['nfs'];
        break;
      case 'NFS':
        delete filteredVolumes[idx]['config_map'];
        delete filteredVolumes[idx]['secret'];
        delete filteredVolumes[idx]['host_path'];
        break;
    }
    delete filteredVolumes[idx]['errors'];
  });
  return filteredVolumes;
};

const checkVolumeErrors = (volumes, setHasErrors) => {
  const volumesCopy = structuredClone(volumes);
  let errorsExist = false;

  volumesCopy.map((volume) => {
    if (!volume['errors']) {
      volume['errors'] = {};
    }
    if (!volume.name) {
      volume.errors['name'] = 'Required';
      errorsExist = true;
    } else if (volume.errors['name']) {
      delete volume.errors['name'];
    }
    if (!volume.archetype) {
      volume.errors['archetype'] = 'Required';
      errorsExist = true;
    } else if (volume.errors['archetype']) {
      delete volume.errors['archetype'];
    }
    if (!volume.mount_path) {
      volume.errors['mount_path'] = 'Required';
      errorsExist = true;
    } else if (volume.errors['mount_path']) {
      delete volume.errors['mount_path'];
    }

    switch (volume['archetype']) {
      case 'ConfigMap':
        break;
      case 'Secret':
        break;
      case 'HostPath':
        if (volume['host_path'] && !volume['host_path']['path']) {
          volume.errors['host_path'] = {};
          volume.errors.host_path['path'] = 'Required';
          errorsExist = true;
        } else if (volume.errors['host_path'] && volume.errors.host_path['path']) {
          delete volume.errors.host_path['path'];
        }
        break;
      case 'NFS':
        volume.errors['nfs'] = {};
        if (volume['nfs'] && !volume.nfs['server']) {
          volume.errors.nfs['server'] = 'Required';
          errorsExist = true;
        } else if (volume.errors['nfs'] && volume.errors.nfs['server']) {
          delete volume.errors.nfs['server'];
        }
        if (volume['nfs'] && !volume.nfs['path']) {
          volume.errors.nfs['path'] = 'Required';
          errorsExist = true;
        } else if (volume.errors['nfs'] && volume.errors.nfs['path']) {
          delete volume.errors.nfs['path'];
        }
        break;
    }
  });

  setHasErrors(errorsExist);
  return volumesCopy;
};

const updateCreateRequestVolumes = (newVolumes, setRequestVolumes) => {
  const requestVolumes = filterVolumeFields(newVolumes);
  setRequestVolumes(requestVolumes);
};

const updateEditRequestVolumes = (volumes, newVolumes, setRequestVolumes) => {
  const filteredVolumes = filterVolumeFields(newVolumes);
  filteredVolumes.map((volume) => {
    // host_path cannot be blank, remove if not specified
    if (volume.archetype == 'HostPath' && volume['host_path'] && volume.host_path.path_type == '') {
      delete volume.host_path.path_type;
      // default_mode can't be blank for ConfigMap or Secret archetypes
    } else if (volume.archetype == 'ConfigMap' && volume['config_map'] && volume.config_map.default_mode == '') {
      delete volume.config_map.default_mode;
      // default_mode can't be blank for ConfigMap or Secret archetypes
    } else if (volume.archetype == 'Secret' && volume['secret'] && volume.secret.default_mode == '') {
      delete volume.secret.default_mode;
    }
  });

  const originalVolumes = [];
  volumes.map((volume) => {
    // push all initial volumes into a remove list
    if (Object(volume).hasOwnProperty('name')) {
      originalVolumes.push(volume.name);
    }
  });

  // build volumes structure for parent
  const requestVolumes = {
    add_volumes: filteredVolumes,
    remove_volumes: originalVolumes,
  };
  setRequestVolumes(requestVolumes);
};

const VolumeInputs = ({ initialVolumes, updateRequestVolumes, setHasErrors, disabled }) => {
  const [volumes, setVolumes] = useState(structuredClone(initialVolumes));

  // update output collection structure
  const updateVolume = (idx, key, subkey, value) => {
    // make a deep copy of the outputCollection
    let volumesCopy = structuredClone(volumes);
    // set the new value for the key
    if (subkey) volumesCopy[idx][key][subkey] = value;
    else volumesCopy[idx][key] = value;

    // updating the archetype may require adding a new key
    if (key == 'archetype' && value == 'ConfigMap' && !Object(volumesCopy[idx]).hasOwnProperty('config_map')) {
      volumesCopy[idx].config_map = { default_mode: '', optional: false };
    }
    if (key == 'archetype' && value == 'Secret' && !Object(volumesCopy[idx]).hasOwnProperty('secret')) {
      volumesCopy[idx].secret = { default_mode: '', optional: false };
    }
    if (key == 'archetype' && value == 'NFS' && !Object(volumesCopy[idx]).hasOwnProperty('nfs')) {
      volumesCopy[idx].nfs = { path: '', server: '' };
    }
    if (key == 'archetype' && value == 'HostPath' && !Object(volumesCopy[idx]).hasOwnProperty('host_path')) {
      volumesCopy[idx].host_path = { path: '', path_type: '' };
    }
    // update the outputCollection object and trigger dom refresh
    volumesCopy = checkVolumeErrors(volumesCopy, setHasErrors);
    setVolumes(volumesCopy);
    updateRequestVolumes(volumesCopy);
  };

  // handle adding a volume to the list
  const addVolume = () => {
    const updatedVolumes = [...volumes, structuredClone(NewVolumeTemplate)];
    // new volumes always have missing fields
    setHasErrors(true);
    setVolumes(updatedVolumes);
  };

  // handle removing a volume from the list
  const removeVolume = (removeIndex) => {
    // remove volume at the specified index
    const updatedVolumes = volumes.filter(function (volume, volumeIndex) {
      if (volumeIndex != removeIndex) {
        return true;
      }
      return false;
    });
    // removing volumes might remove the last remaining error, check for updated errors
    const volumesCopy = checkVolumeErrors(updatedVolumes, setHasErrors);
    setVolumes(volumesCopy);
    updateRequestVolumes(volumesCopy);
  };

  // this is needed for onload when cloning from an existing image
  useEffect(() => {
    // we don't check for errors because we assume the source for copy is valid and
    // might want to not assume that
    updateRequestVolumes(initialVolumes);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Fragment>
      {volumes &&
        volumes.length > 0 &&
        !disabled &&
        volumes.map((volume, idx) => (
          <Row key={idx} className="my-2 image-fields">
            <Col>
              <Row>
                <Col className="selectable-img-col">
                  <Subtitle>Name</Subtitle>
                </Col>
                <Col className="m-2">
                  <OverlayTipRight tip={VolumesToolTips.name}>
                    <Form.Control
                      type="textarea"
                      placeholder={'name'}
                      value={volume.name}
                      onChange={(e) => updateVolume(idx, 'name', '', String(e.target.value))}
                    />
                  </OverlayTipRight>
                  {volume['errors'] && 'name' in volume['errors'] && (
                    <Alert variant="danger" className="d-flex justify-content-center m-2">
                      {volume['errors'].name}
                    </Alert>
                  )}
                </Col>
              </Row>
              <Row>
                <Col className="selectable-img-col">
                  <Subtitle>Archetype</Subtitle>
                </Col>
                <Col className="m-2">
                  <OverlayTipRight tip={VolumesToolTips.archetype}>
                    <Form.Select value={volume.archetype} onChange={(e) => updateVolume(idx, 'archetype', '', String(e.target.value))}>
                      {!volume.archetype && <option>Select an Archetype</option>}
                      {DisplayArchetypes.map((archetype) => (
                        <option key={archetype} value={archetype}>
                          {archetype}
                        </option>
                      ))}
                    </Form.Select>
                  </OverlayTipRight>
                  {volume['errors'] && 'archetype' in volume['errors'] && (
                    <Alert variant="danger" className="d-flex justify-content-center m-2">
                      {volume['errors'].archetype}
                    </Alert>
                  )}
                </Col>
              </Row>
              {(volume['archetype'] == 'ConfigMap' || volume['archetype'] == 'Secret') && (
                <Fragment>
                  <Row>
                    <Col className="selectable-img-col">
                      <Subtitle>Default Mode (octal)</Subtitle>
                    </Col>
                    <Col className="m-2">
                      <OverlayTipRight tip={VolumesToolTips.mode}>
                        <Form.Control
                          type="textarea"
                          placeholder={600}
                          value={volume[volume['archetype'] == 'ConfigMap' ? 'config_map' : 'secret']['default_mode']}
                          onChange={(e) => {
                            if (!isNaN(e.target.value)) {
                              updateVolume(
                                idx,
                                volume['archetype'] == 'ConfigMap' ? 'config_map' : 'secret',
                                'default_mode',
                                e.target.value == '' ? '' : Number(e.target.value.replace(/[^0-9]+/gi, '')),
                              );
                            }
                          }}
                        />
                      </OverlayTipRight>
                    </Col>
                  </Row>
                  <Row>
                    <Col className="selectable-img-col">
                      <Subtitle>Optional</Subtitle>
                    </Col>
                    <Col className="m-2">
                      <OverlayTipRight tip={VolumesToolTips.optional}>
                        <h6>
                          <Form.Check
                            type="switch"
                            label=""
                            checked={volume[volume['archetype'] == 'ConfigMap' ? 'config_map' : 'secret']['optional']}
                            onChange={(e) =>
                              updateVolume(
                                idx,
                                volume['archetype'] == 'ConfigMap' ? 'config_map' : 'secret',
                                'optional',
                                !volume[volume['archetype'] == 'ConfigMap' ? 'config_map' : 'secret']['optional'],
                              )
                            }
                          />
                        </h6>
                      </OverlayTipRight>
                    </Col>
                  </Row>
                </Fragment>
              )}
              {volume['archetype'] == 'HostPath' && (
                <Fragment>
                  <Row>
                    <Col className="selectable-img-col">
                      <Subtitle>Path</Subtitle>
                    </Col>
                    <Col className="m-2">
                      <OverlayTipRight tip={VolumesToolTips.host_path}>
                        <Form.Control
                          type="textarea"
                          placeholder={'/host/src/path'}
                          value={'host_path' in volume ? volume['host_path']['path'] : ''}
                          onChange={(e) => updateVolume(idx, 'host_path', 'path', String(e.target.value))}
                        />
                      </OverlayTipRight>
                      {volume['errors'] && 'host_path' in volume.errors && 'path' in volume.errors.host_path && (
                        <Alert variant="danger" className="d-flex justify-content-center m-2">
                          {volume.errors.host_path.path}
                        </Alert>
                      )}
                    </Col>
                  </Row>
                  <Row>
                    <Col className="selectable-img-col">
                      <Subtitle>Path Type</Subtitle>
                    </Col>
                    <Col className="m-2">
                      <OverlayTipRight tip={VolumesToolTips.host_path_type}>
                        <Form.Select onChange={(e) => updateVolume(idx, 'host_path', 'path_type', String(e.target.value))}>
                          <option value="">Select a Path Type</option>
                          {HostPathTypes.map((pathType) => (
                            <option key={pathType} value={pathType}>
                              {pathType}
                            </option>
                          ))}
                        </Form.Select>
                      </OverlayTipRight>
                    </Col>
                  </Row>
                </Fragment>
              )}
              {volume['archetype'] == 'NFS' && (
                <Fragment>
                  <Row>
                    <Col className="selectable-img-col">
                      <Subtitle>Server</Subtitle>
                    </Col>
                    <Col className="m-2">
                      <OverlayTipRight tip={VolumesToolTips.nfs_server}>
                        <Form.Control
                          type="textarea"
                          placeholder={'hostname'}
                          value={'nfs' in volume ? volume['nfs']['server'] : ''}
                          onChange={(e) => updateVolume(idx, 'nfs', 'server', String(e.target.value))}
                        />
                      </OverlayTipRight>
                      {volume['errors'] && 'nfs' in volume.errors && 'server' in volume.errors.nfs && (
                        <Alert variant="danger" className="d-flex justify-content-center m-2">
                          {volume.errors.nfs.server}
                        </Alert>
                      )}
                    </Col>
                  </Row>
                  <Row>
                    <Col className="selectable-img-col">
                      <Subtitle>Path</Subtitle>
                    </Col>
                    <Col className="m-2">
                      <OverlayTipRight tip={VolumesToolTips.nfs_path}>
                        <Form.Control
                          type="textarea"
                          placeholder={'/path/to/directory'}
                          value={'nfs' in volume ? volume['nfs']['path'] : ''}
                          onChange={(e) => updateVolume(idx, 'nfs', 'path', String(e.target.value))}
                        />
                      </OverlayTipRight>
                      {volume['errors'] && 'nfs' in volume.errors && 'path' in volume.errors.nfs && (
                        <Alert variant="danger" className="d-flex justify-content-center m-2">
                          {volume.errors.nfs.path}
                        </Alert>
                      )}
                    </Col>
                  </Row>
                </Fragment>
              )}
              <Row>
                <Col className="selectable-img-col">
                  <Subtitle>Mount Path</Subtitle>
                </Col>
                <Col className="m-2">
                  <OverlayTipRight tip={VolumesToolTips.mount_path}>
                    <Form.Control
                      type="textarea"
                      placeholder={'mount path'}
                      value={volume['mount_path']}
                      onChange={(e) => updateVolume(idx, 'mount_path', '', String(e.target.value))}
                    />
                  </OverlayTipRight>
                  {volume['errors'] && 'mount_path' in volume.errors && (
                    <Alert variant="danger" className="d-flex justify-content-center m-2">
                      {volume.errors.mount_path}
                    </Alert>
                  )}
                </Col>
              </Row>
              <Row>
                <Col className="selectable-img-col">
                  <Subtitle>Sub Path</Subtitle>
                </Col>
                <Col className="m-2">
                  <OverlayTipRight tip={VolumesToolTips.sub_path}>
                    <Form.Control
                      type="textarea"
                      placeholder={'sub path'}
                      value={volume['sub_path']}
                      onChange={(e) => updateVolume(idx, 'sub_path', '', String(e.target.value))}
                    />
                  </OverlayTipRight>
                </Col>
              </Row>
              <Row>
                <Col className="selectable-img-col">
                  <Subtitle>Read Only</Subtitle>
                </Col>
                <Col className="m-2">
                  <OverlayTipRight tip={VolumesToolTips.read_only}>
                    <h6>
                      <Form.Check
                        type="switch"
                        id={idx + '-volume-read-only'}
                        label=""
                        checked={volume['read_only']}
                        onChange={(e) => updateVolume(idx, 'read_only', '', !volume['read_only'])}
                      />
                    </h6>
                  </OverlayTipRight>
                </Col>
              </Row>
              <Row>
                <Col className="selectable-img-col">
                  <Subtitle>Kustomize</Subtitle>
                </Col>
                <Col className="m-2">
                  <OverlayTipRight tip={VolumesToolTips.kustomize}>
                    <h6>
                      <Form.Check
                        type="switch"
                        label=""
                        checked={volume['kustomize']}
                        onChange={(e) => updateVolume(idx, 'kustomize', '', !volume['kustomize'])}
                      />
                    </h6>
                  </OverlayTipRight>
                </Col>
              </Row>
            </Col>
            <Col className="d-flex justify-content-center align-items-center" style={{ maxWidth: '60px' }}>
              <Button className="danger-btn" onClick={() => removeVolume(idx)}>
                <FaTrash />
              </Button>
            </Col>
            <hr />
          </Row>
        ))}
      {/* Add volume button should be shown only for k8s scaled images */}
      {volumes && !disabled && (
        <Row>
          <Col>
            <Button className="ok-btn" disabled={disabled} onClick={() => addVolume()}>
              <b>+</b>
            </Button>
          </Col>
        </Row>
      )}
    </Fragment>
  );
};

const EditVolumesFields = ({ initialVolumes, updateRequestVolumes, setHasErrors, disabled }) => {
  return (
    <Row>
      <Col style={{ flex: 0.2 }}></Col>
      <Col style={{ flex: 1.25 }}></Col>
      <Col style={{ flex: 8 }}>
        <VolumeInputs
          initialVolumes={initialVolumes}
          updateRequestVolumes={updateRequestVolumes}
          setHasErrors={setHasErrors}
          disabled={disabled}
        />
      </Col>
    </Row>
  );
};

const CreateVolumesFields = ({ initialVolumes, updateRequestVolumes, setHasErrors, disabled }) => {
  return (
    <Row>
      <Col className="title-col">
        <h5>Volumes</h5>
      </Col>
      <Col className="field-col">
        <VolumeInputs
          initialVolumes={initialVolumes}
          updateRequestVolumes={updateRequestVolumes}
          setHasErrors={setHasErrors}
          disabled={disabled}
        />
      </Col>
    </Row>
  );
};

const ImageVolumes = ({ volumes, setRequestVolumes, mode, setHasErrors, disabled }) => {
  // provide the edit/create components with a callback to update a
  // request formatted volumes object
  const setUpdatedVolumes = (newVolumes) => {
    if (['Create', 'Copy'].includes(mode)) {
      return updateCreateRequestVolumes(newVolumes, setRequestVolumes);
    } else {
      return updateEditRequestVolumes(volumes, newVolumes, setRequestVolumes);
    }
  };

  // add empty error key to initial volumes for local tracking of missing or invalid fields
  volumes.map((volume) => {
    volume['errors'] = {};
  });

  // copy mode requires cleanup of the duplicated image configuration before being placed into
  // the create component
  if (mode == 'Copy') {
    return (
      <CreateVolumesFields
        initialVolumes={volumes}
        updateRequestVolumes={setUpdatedVolumes}
        setHasErrors={setHasErrors}
        disabled={disabled}
      />
    );
  } else if (mode == 'Create') {
    return (
      <CreateVolumesFields initialVolumes={[]} updateRequestVolumes={setUpdatedVolumes} setHasErrors={setHasErrors} disabled={disabled} />
    );
  }

  return (
    <Fragment>
      <Row>
        <Col>
          <OverlayTipRight tip={VolumesToolTips.self}>
            <b>{'Volumes'}</b> <FaQuestionCircle />
          </OverlayTipRight>
        </Col>
      </Row>
      {mode == 'View' && <DisplayVolumes volumes={volumes} />}
      {mode == 'Edit' && (
        <EditVolumesFields
          initialVolumes={volumes}
          updateRequestVolumes={setUpdatedVolumes}
          setHasErrors={setHasErrors}
          disabled={disabled}
        />
      )}
    </Fragment>
  );
};

export default ImageVolumes;
