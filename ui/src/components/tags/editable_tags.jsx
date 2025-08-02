import { useState, useEffect } from 'react';
import { ButtonToolbar, Button, Card, Col, Modal, Row } from 'react-bootstrap';
import { FaBackspace, FaRegEdit, FaSave } from 'react-icons/fa';
import Select from 'react-select';
import CreatableSelect from 'react-select/creatable';

// project imports
import { AlertBanner, OverlayTipBottom, OverlayTipRight, Subtitle } from '@components';
import { createReactSelectStyles } from '@utilities';
import { FormattedFileInfoTagKeys, TLPLevels } from './utilities';
import { TagBadge } from './tags';
import { deleteTags, getFileDetails, uploadTags } from '@thorpi';
import rawAttackTagDefaults from '../../../mitre_tags/attackTagsList.tags?raw';
import rawMbcTagDefaults from '../../../mitre_tags/MBCTagsList.tags?raw';

// styles for react select badges
const tlpTagStyle = createReactSelectStyles('White', 'rgb(160, 162, 163)');
const generalTagStyle = createReactSelectStyles('White', 'rgb(160, 162, 163)');
const fileInfoTagStyles = createReactSelectStyles('White', '#7ba8ec');
const mitreTagStyles = createReactSelectStyles('White', 'rgb(227, 135, 81)');

const EditableTags = ({ sha256, tags, setDetails, setUpdateError, screenWidth }) => {
  const [editing, setEditing] = useState(false);
  const [pendingTags, setPendingTags] = useState({});
  const [deletedTags, setDeletedTags] = useState({});
  const [invalidPendingTags, setInvalidPendingTags] = useState([]);
  const [selectedGeneralTags, setSelectedGeneralTags] = useState([]);
  const [generalTagOptions, setGeneralTagOptions] = useState([]);
  const [selectedFileInfoTags, setSelectedFileInfoTags] = useState([]);
  const [fileInfoTagOptions, setFileInfoTagOptions] = useState([]);
  const [selectedTlpTags, setSelectedTlpTags] = useState([]);
  const [selectedMBCTags, setSelectedMBCTags] = useState([]);
  const [selectedAttackTags, setSelectedAttackTags] = useState([]);
  const [showUpdateModal, setShowUpdateModal] = useState(false);
  const [deleteErrorStatus, setDeleteErrorStatus] = useState('');
  const [createErrorStatus, setCreateErrorStatus] = useState('');
  const [numberOfChanges, setNumberOfChanges] = useState(0);

  // build a list of tags for non-general tag sections
  const excludeTags = [...FormattedFileInfoTagKeys, 'TLP', 'RESULTS', 'ATT&CK', 'MBC'];

  // calculate the number of tags using unique values
  let tagsCount = 0;
  if (tags && Object.keys(tags).length > 0) {
    for (const tag in tags) {
      if (Object.prototype.hasOwnProperty.call(tags, tag)) {
        tagsCount += Object.keys(tags[tag]).length;
      }
    }
  }

  // get TLP tags and React Select formatted options
  const tlpTagOptions = TLPLevels.sort().map((tagValue) => ({
    value: tagValue,
    label: tagValue,
    thoriumTag: { key: 'TLP', value: tagValue },
  }));
  const tlpTags = !(tags && tags.TLP)
    ? []
    : Object.keys(tags['TLP'])
        .sort()
        .map((tagValue) => {
          return tagValue;
        });
  const initialSelectedTLPTags = tlpTags.map((tagValue) => {
    return {
      value: tagValue,
      label: tagValue,
      thoriumTag: { key: 'TLP', value: tagValue },
    };
  });

  const compareEntryKeys = (arrayObj1, arrayObj2) => {
    if (Array.isArray(arrayObj1) && Array.isArray(arrayObj2)) {
      if (arrayObj1[0] > arrayObj2[0]) return 1;
      else if (arrayObj1[0] < arrayObj2[0]) return -1;
    }
    // otherwise assume equal or incomparable
    return 0;
  };

  // get General tags and React Select formatted options
  const generalTags = Object.fromEntries(Object.entries(tags).filter(([k]) => !excludeTags.includes(k.toUpperCase())));
  const initialGeneralTagOptions = [];
  Object.keys(generalTags)
    .sort()
    .map((tagKey) => {
      if (!excludeTags.includes(tagKey)) {
        Object.keys(generalTags[tagKey])
          .sort()
          .map((tagValue) => {
            const tag = `${tagKey}: ${tagValue}`;
            initialGeneralTagOptions.push({
              value: tag,
              label: tag,
              thoriumTag: { key: tagKey, value: tagValue },
            });
          });
      }
    });

  // get File Info tags and React Select formatted options
  const fileInfoTags = Object.fromEntries(Object.entries(tags).filter(([k, v]) => FormattedFileInfoTagKeys.includes(k.toUpperCase())));
  const initialFileInfoTagOptions = [];

  Object.entries(fileInfoTags).map(([tagKey]) => {
    if (FormattedFileInfoTagKeys.includes(tagKey.toUpperCase())) {
      Object.entries(tags[tagKey])
        .sort(compareEntryKeys)
        .map(([tagValue]) => {
          const tag = `${tagKey}: ${tagValue}`;
          initialFileInfoTagOptions.push({
            value: tag,
            label: tag,
            thoriumTag: { key: tagKey, value: tagValue },
          });
        });
    }
  });

  // get a list of all MBC tags for this file
  let mbcTags = [];
  Object.entries(tags).filter(([k]) => {
    if (k.toUpperCase() == 'MBC') {
      mbcTags.push(...Object.keys(tags[k]));
    }
  });
  mbcTags = mbcTags.sort();

  // create a ReactSelect formatted list of current MBC tags
  const initialMBCTags = [];
  mbcTags.map((mbcValue) => {
    initialMBCTags.push({
      value: mbcValue,
      label: mbcValue,
      thoriumTag: { key: 'MBC', value: mbcValue },
    });
  });

  // create a list of all possible MBC tags
  const mbcTagOptions = String(rawMbcTagDefaults)
    .split('\n')
    .map((mbcValue) => {
      return {
        value: mbcValue,
        label: mbcValue,
        thoriumTag: { key: 'MBC', value: mbcValue },
      };
    });

  // get a list of all ATT&CK tags for this file
  let attackTags = [];
  Object.entries(tags).filter(([k]) => {
    if (k.toUpperCase() == 'ATT&CK') {
      attackTags.push(...Object.keys(tags[k]));
    }
  });
  attackTags = attackTags.sort();

  // create a ReactSelect formatted list of current ATT&CK tags
  const initialAttackTags = [];
  attackTags.map((attackValue) => {
    initialAttackTags.push({
      value: attackValue,
      label: attackValue,
      thoriumTag: { key: 'ATT&CK', value: attackValue },
    });
  });

  // create a list of all possible ATT&CK tags
  const attackTagOptions = String(rawAttackTagDefaults)
    .split('\n')
    .map((attackValue) => {
      return {
        value: attackValue,
        label: attackValue,
        thoriumTag: { key: 'ATT&CK', value: attackValue },
      };
    });

  // clear out any pending tag deletions or additions
  const resetTags = () => {
    setDeleteErrorStatus('');
    setCreateErrorStatus('');
    // clear tag change state
    setPendingTags({});
    setDeletedTags({});
    setInvalidPendingTags([]);
    setNumberOfChanges(0);

    // reset TLP tags
    setSelectedTlpTags(structuredClone(initialSelectedTLPTags));
    // General tags
    const resetGeneralTags = structuredClone(initialGeneralTagOptions);
    setSelectedGeneralTags(resetGeneralTags);
    setGeneralTagOptions(resetGeneralTags);
    // reset File Info tags
    const resetFileInfoTags = structuredClone(initialFileInfoTagOptions);
    setFileInfoTagOptions(resetFileInfoTags);
    setSelectedFileInfoTags(resetFileInfoTags);
    // reset Mitre MBC tags
    const resetMBCTags = structuredClone(initialMBCTags);
    setSelectedMBCTags(resetMBCTags);
    // reset Mitre Att&ck tags
    const resetAttackTags = structuredClone(initialAttackTags);
    setSelectedAttackTags(resetAttackTags);
  };

  // Commit tag changes to APIs
  // This submits new tags and deletes tags based on pending user inputs
  const commitTagUpdates = async () => {
    setDeleteErrorStatus('');
    setCreateErrorStatus('');
    // always delete tags first to ensure added tags aren't lost
    // when a tag was added and deleted in the same commit (which
    // should already be handled with guard rails)
    let error = false;
    if (deletedTags && Object.keys(deletedTags).length > 0) {
      const data = { tags: deletedTags };
      const deleteSuccess = await deleteTags(sha256, data, setDeleteErrorStatus);
      if (!deleteSuccess) {
        error = true;
      }
    }

    if (Object.keys(pendingTags).length > 0) {
      const data = { tags: pendingTags };
      const createSuccess = await uploadTags(sha256, data, setCreateErrorStatus);
      if (!createSuccess) {
        error = true;
      }
    }

    if (!error) {
      // update file details (specifically tags) when tags have been updated
      const updatedFileDetails = await getFileDetails(sha256, setUpdateError);
      if (updatedFileDetails) {
        setShowUpdateModal(false);
        setDetails(updatedFileDetails);
        // exit editing mode after changes are committed
        setEditing(false);
      }
    }
  };

  // get a count of uncommitted tags changes
  const updateUpdatedTagCount = (pending, deleted, invalid) => {
    let count = 0;
    Object.keys(pending).map((tag) => {
      count += Object.keys(pending[tag]).length;
    });
    Object.keys(deleted).map((tag) => {
      count += Object.keys(deleted[tag]).length;
    });
    count += invalid.length;
    setNumberOfChanges(count);
  };

  // Determine pending and deleted tags from React select set selectedTags
  const updateSelectedTags = (newValue) => {
    const updatedPendingTags = structuredClone(pendingTags);
    const updatedDeletedTags = structuredClone(deletedTags);
    let updatedInvalidPendingTags = structuredClone(invalidPendingTags);

    // clear is an action that select can enable with the isClearable property
    // this property can be disabled with isClearable={false}
    if (newValue.action == 'clear') {
      // check if 'removedValues' matches non-pending tags
      const cleared = newValue['removedValues'];
      if (cleared && cleared.length > 0) {
        cleared.map((selectTag) => {
          // grab formatted thorium key/value tag object
          const tag = selectTag.thoriumTag;
          // check if cleared tag is in pending object
          if (updatedPendingTags[tag.key] && updatedPendingTags[tag.key].includes(tag.value)) {
            // we found the object, now remove it from list of values for this key
            if (updatedPendingTags[tag.key].length > 1) {
              updatedPendingTags[tag.key] = updatedPendingTags[tag.key].filter((value) => {
                return value != tag.value;
              });
              // or remove the whole key/value tag array since there is one or less tags for this key
              // technically empty shouldn't be valid but it will be caught here
            } else {
              delete updatedPendingTags[tag.key];
            }
            // tag was invalid when added, it only needs to get removed from the invalid list
          } else if (tag.key && !tag.value && updatedInvalidPendingTags.includes(tag.key)) {
            updatedInvalidPendingTags = updatedInvalidPendingTags.filter((invalidKey) => {
              return invalidKey != tag.key;
            });
            // add existing Thorium tag to delete object for removal
            // this mean tag wasn't a pending tag and wasn't invalid (has key + value)
          } else {
            if (tag.key in updatedDeletedTags) {
              updatedDeletedTags[tag.key].push(tag.value);
            } else {
              updatedDeletedTags[tag.key] = [tag.value];
            }
          }
          // if not in pending, add to deleted
        });
      }
    } else if (newValue.action == 'remove-value') {
      const tag = newValue.removedValue.thoriumTag;
      // remove tag from pending since this is a new unsaved tag
      if (tag.key in updatedPendingTags && updatedPendingTags[tag.key].includes(tag.value)) {
        // we found the object, now remove it from list of values for this key
        if (updatedPendingTags[tag.key].length > 1) {
          updatedPendingTags[tag.key] = updatedPendingTags[tag.key].filter((value) => {
            return value != tag.value;
          });
          // or remove the whole key/value tag array since there is one or less tags for this key
          // technically empty shouldn't be valid but it will be caught here
        } else {
          delete updatedPendingTags[tag.key];
        }
        // removing pending invalid tag from invalid list
      } else if (tag.key && !tag.value && updatedInvalidPendingTags.includes(tag.key)) {
        // filter list and update the invalid pending tag list
        updatedInvalidPendingTags = updatedInvalidPendingTags.filter((invalidKey) => {
          return invalidKey != tag.key;
        });
        // add existing Thorium tag to delete object for removal
        // this mean tag wasn't a pending tag and wasn't invalid (has key + value)
      } else {
        if (tag.key in updatedDeletedTags) {
          updatedDeletedTags[tag.key].push(tag.value);
        } else {
          updatedDeletedTags[tag.key] = [tag.value];
        }
      }
    } else if (newValue.action == 'select-option') {
      const tag = newValue.option.thoriumTag;
      // check if tag was set to be deleted; this only happens
      // when a tag already existed not when its being created
      if (tag.key in updatedDeletedTags && updatedDeletedTags[tag.key].includes(tag.value)) {
        if (updatedDeletedTags[tag.key].length > 1) {
          updatedDeletedTags[tag.key] = updatedDeletedTags[tag.key].filter((value) => {
            return value != tag.value;
          });
        } else {
          delete updatedDeletedTags[tag.key];
        }
      } else {
        if (tag.key in updatedPendingTags) {
          updatedPendingTags[tag.key].push(tag.value);
        } else {
          updatedPendingTags[tag.key] = [tag.value];
        }
      }
    } else if (newValue.action == 'create-option') {
      // parse tag based on '=' or ':' delimiter
      const parseRawTag = (rawTag) => {
        if (rawTag.includes(':')) {
          const tagKey = rawTag.split(':', 1)[0];
          return { key: tagKey, value: rawTag.substring(tagKey.length + 1) };
        } else if (rawTag.includes('=')) {
          const tagKey = rawTag.split('=', 1)[0];
          return { key: tagKey, value: rawTag.substring(tagKey.length + 1) };
        } else {
          return { key: rawTag, value: null };
        }
      };

      const tag = parseRawTag(newValue.option.value);
      // tag is not valid
      if (tag.key && tag.value) {
        // add thorium key/value info to selected tag structure
        // this only works the option member field is a shallow copy of
        // whats stored in the selectedTags structure for the value that is
        // used by the Select and CreatableSelect components
        newValue.option['thoriumTag'] = tag;
        if (tag.key in updatedPendingTags) {
          updatedPendingTags[tag.key].push(tag.value);
        } else {
          updatedPendingTags[tag.key] = [tag.value];
        }
        // tag is invalid, it must have a distinct key and value separated
        // by a colon delimiter
      } else {
        newValue.option['thoriumTag'] = tag;
        if (!updatedInvalidPendingTags.includes(tag.key)) {
          updatedInvalidPendingTags.push(tag.key);
        }
      }
    }
    // update tag counts and save pending/deleted tags to react state
    updateUpdatedTagCount(updatedPendingTags, updatedDeletedTags, updatedInvalidPendingTags);
    setDeletedTags(updatedDeletedTags);
    setPendingTags(updatedPendingTags);
    setInvalidPendingTags(updatedInvalidPendingTags);
    return;
  };

  // on rerender, update
  useEffect(() => {
    if (tags) {
      resetTags();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tags]);

  return (
    <Card className="panel ps-4">
      <Card.Body>
        <Row className="align-items-center">
          <Col className="tags-col">
            <Row>
              <Col>
                <Subtitle>TLP</Subtitle>
              </Col>
              <Col className="details-tags-name">
                {editing ? (
                  <Select
                    isMulti
                    isSearchable
                    isClearable
                    value={selectedTlpTags}
                    styles={tlpTagStyle}
                    onChange={(selected, newValue) => {
                      setSelectedTlpTags(selected);
                      updateSelectedTags(newValue);
                    }}
                    options={tlpTagOptions}
                  />
                ) : (
                  tlpTags.map((level) => (
                    <TagBadge resource="file" key={level} tag={'TLP'} value={level} condensed={false} action={'link'} />
                  ))
                )}
              </Col>
            </Row>
            <Row>
              <hr className="tagshr" />
              <Col>
                <Subtitle>Tags</Subtitle>
              </Col>
              <Col className="details-tags-name">
                {editing ? (
                  <CreatableSelect
                    isMulti
                    isSearchable
                    isClearable
                    value={selectedGeneralTags}
                    styles={generalTagStyle}
                    noOptionsMessage={({ inputValue }) =>
                      inputValue
                        ? inputValue
                        : `Create a tag by
                      typing in a key and value separated by = or : and then clicking enter.`
                    }
                    onChange={(selected, newValue) => {
                      setSelectedGeneralTags(selected);
                      updateSelectedTags(newValue);
                    }}
                    options={generalTagOptions}
                  />
                ) : (
                  Object.keys(generalTags)
                    .sort()
                    .map((tagKey) =>
                      Object.keys(generalTags[tagKey])
                        .sort()
                        .map((tagValue, idx) => (
                          <TagBadge resource="file" key={idx} tag={tagKey} value={tagValue} condensed={false} action={'link'} />
                        )),
                    )
                )}
              </Col>
            </Row>
            <Row>
              <hr className="tagshr" />
              <Col>
                <Subtitle>File Info</Subtitle>
              </Col>
              <Col className="details-tags-name">
                {editing ? (
                  <CreatableSelect
                    isMulti
                    isSearchable
                    isClearable
                    value={selectedFileInfoTags}
                    styles={fileInfoTagStyles}
                    noOptionsMessage={({ inputValue }) =>
                      inputValue
                        ? inputValue
                        : `Create a tag by
                      typing in a key and value separated by = or : and then clicking enter.
                      Tags in the FILE INFO section are regular tags with keys such as FileType,
                      Compiler, and Arch. These tags are displayed in their own section to make
                      them easier to find.`
                    }
                    onChange={(selected, newValue) => {
                      setSelectedFileInfoTags(selected);
                      updateSelectedTags(newValue);
                    }}
                    options={fileInfoTagOptions}
                  />
                ) : (
                  Object.keys(fileInfoTags)
                    .sort()
                    .map((tagKey) =>
                      Object.keys(fileInfoTags[tagKey])
                        .sort()
                        .map((tagValue, idx) => (
                          <TagBadge resource="file" key={idx} tag={tagKey} value={tagValue} condensed={false} action={'link'} />
                        )),
                    )
                )}
              </Col>
            </Row>
            <Row>
              <hr className="tagshr" />
              <Col>
                <Subtitle>{`ATT&CK`}</Subtitle>
              </Col>
              <Col className="details-tags-name">
                {editing ? (
                  <Select
                    isMulti
                    isSearchable
                    isClearable
                    value={selectedAttackTags}
                    styles={mitreTagStyles}
                    onChange={(selected, newValue) => {
                      setSelectedAttackTags(selected);
                      updateSelectedTags(newValue);
                    }}
                    options={attackTagOptions}
                  />
                ) : (
                  attackTags.map((tagValue, idx) => (
                    <TagBadge resource="file" key={tagValue} tag={'ATT&CK'} value={tagValue} condensed={false} action={'docs'} />
                  ))
                )}
              </Col>
            </Row>
            <Row>
              <hr className="tagshr" />
              <Col>
                <Subtitle>{`MBC`}</Subtitle>
              </Col>
              <Col className="details-tags-name">
                {editing ? (
                  <Select
                    isMulti
                    isSearchable
                    isClearable
                    value={selectedMBCTags}
                    styles={mitreTagStyles}
                    onChange={(selected, newValue) => {
                      setSelectedMBCTags(selected);
                      updateSelectedTags(newValue);
                    }}
                    options={mbcTagOptions}
                  />
                ) : (
                  mbcTags.map((tagValue, idx) => (
                    <TagBadge resource="file" key={tagValue} tag={'MBC'} value={tagValue} condensed={false} action={'docs'} />
                  ))
                )}
              </Col>
            </Row>
          </Col>
          <Col xs={1} className="edit-icon">
            <div className="edit-icon left-edit-tag-btn d-flex justify-content-center">
              <EditTagButton
                editing={editing}
                setEditing={setEditing}
                numberOfChanges={numberOfChanges}
                setShowUpdateModal={setShowUpdateModal}
                resetTags={resetTags}
              />
            </div>
          </Col>
        </Row>
        <Modal show={showUpdateModal} onHide={() => setShowUpdateModal(false)} backdrop="static" keyboard={false}>
          <Modal.Header closeButton>
            <Modal.Title>Confirm Tags Changes?</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            {Object.keys(pendingTags).length > 0 && (
              <Row>
                <Col>
                  <b>Add: </b>
                  {Object.keys(pendingTags).map((tag) =>
                    pendingTags[tag].map((value) => (
                      <TagBadge
                        resource="file"
                        key={tag + '_' + value}
                        tag={tag}
                        value={tag.toUpperCase() == 'TLP' ? value.toUpperCase() : value}
                        action={'none'}
                        condensed={true}
                      />
                    )),
                  )}
                </Col>
              </Row>
            )}
            {createErrorStatus && (
              <Row className="d-flex justify-content-center p-2">
                <AlertBanner prefix={'Add tags'} errorStatus={createErrorStatus} />
              </Row>
            )}
            {Object.keys(deletedTags).length > 0 && (
              <Row>
                <Col>
                  <b>Delete: </b>
                  {Object.keys(deletedTags).map((tag) =>
                    deletedTags[tag].map((value) => (
                      <TagBadge
                        resource="file"
                        key={tag + '_' + value}
                        tag={tag}
                        value={tag.toUpperCase() == 'TLP' ? value.toUpperCase() : value}
                        action={'none'}
                        condensed={true}
                      />
                    )),
                  )}
                </Col>
              </Row>
            )}
            {deleteErrorStatus && (
              <Row className="d-flex justify-content-center p-2">
                <AlertBanner prefix={'Delete tags'} errorStatus={deleteErrorStatus} />
              </Row>
            )}
            {invalidPendingTags.length > 0 && (
              <Row>
                <Col>
                  <b>Invalid tags: </b>
                  {invalidPendingTags.map((tag) => (
                    <TagBadge resource="file" key={tag} tag={tag} value={''} condensed={true} action={'none'} />
                  ))}
                  <i>
                    (Custom tags must have a key and value that are separated by a colon delimiter. Invalid tags will be ignored when saving
                    other tag changes)`
                  </i>
                </Col>
              </Row>
            )}
          </Modal.Body>
          <Modal.Footer className="d-flex justify-content-center">
            <ButtonToolbar>
              <OverlayTipBottom
                tip={`Cancel pending tag deletions and additions
                and return to the details page`}
              >
                <Button
                  className="primary-btn xsmall-button"
                  onClick={() => {
                    resetTags();
                    setShowUpdateModal(false);
                  }}
                >
                  Clear
                </Button>
              </OverlayTipBottom>
              <OverlayTipBottom tip={`Submit pending tag deletions and additions`}>
                <Button
                  className="warning-btn xsmall-button"
                  disabled={numberOfChanges - invalidPendingTags.length == 0}
                  onClick={() => commitTagUpdates()}
                >
                  Confirm
                </Button>
              </OverlayTipBottom>
            </ButtonToolbar>
          </Modal.Footer>
        </Modal>
      </Card.Body>
    </Card>
  );
};

const EditTagButton = ({ editing, setEditing, numberOfChanges, setShowUpdateModal, resetTags }) => {
  return (
    <ButtonToolbar>
      <OverlayTipRight tip={!editing ? 'Click to add or remove tags.' : 'Click to cancel editing tags'}>
        <Button
          className="icon-btn"
          variant=""
          onClick={() => {
            setEditing(!editing);
            if (editing) {
              resetTags();
            }
          }}
        >
          {editing ? <FaBackspace size="24" /> : <FaRegEdit size="24" />}
        </Button>
      </OverlayTipRight>
      {(editing || numberOfChanges > 0) && (
        <OverlayTipRight
          tip={`There are ${numberOfChanges} pending tag changes.
                  Click to review and submit or cancel pending changes.`}
        >
          <Button className="icon-btn" variant="" disabled={numberOfChanges == 0} onClick={() => setShowUpdateModal(true)}>
            <FaSave size="20" /> {numberOfChanges > 0 && `${numberOfChanges}`}
          </Button>
        </OverlayTipRight>
      )}
    </ButtonToolbar>
  );
};

export default EditableTags;
