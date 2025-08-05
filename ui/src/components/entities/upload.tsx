//@ts-nocheck
import React, { Fragment, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Alert, Button, Card, Col, Container, Form, ProgressBar, Row, Tabs, Tab } from 'react-bootstrap';
import { FaChevronDown, FaChevronUp, FaRedo } from 'react-icons/fa';
import { isIP } from 'is-ip';

// project imports
import {
  OverlayTipTop,
  RunReactionAlerts,
  SelectableArray,
  SelectableDictionary,
  SelectPipelines,
  submitReactions,
  Subtitle,
  UploadDropzone,
  SelectInput,
  SelectInputArray,
} from '@components';
import { useAuth } from '@utilities';
import { uploadFile } from '@thorpi';

const PARALLELUPLOADLIMIT = 5;

const TLPColors = {
  CLEAR: 'tlp-clear',
  GREEN: 'tlp-green',
  AMBER: 'tlp-amber',
  'AMBER+STRICT': 'tlp-amber',
  RED: 'tlp-red',
};

type ProgressBarProps = {
  name: string;
  value: number;
  error: string;
};

// This container allows the progress bar color to change depending on error conditions.
const ProgressBarContainer: React.FC<ProgressBarProps> = ({ name, value, error }) => {
  return (
    <>
      {value < 100 && (
        <ProgressBar animated key={name} label={value} now={value} className={error ? 'warning-bar' : 'info-bar'}></ProgressBar>
      )}
      {value >= 100 && <ProgressBar key={name} label={value} now={value} className={error ? 'danger-bar' : 'success-bar'}></ProgressBar>}
    </>
  );
};

type UploadProps = {
  entity: Entity | undefined;
};

export const Upload: React.FC<UploadProps> = ({ entity }) => {
  const { userInfo } = useAuth();
  const [filesArray, setFilesArray] = useState([]);
  const [description, setDescription] = useState('');
  const [originType, setOriginType] = useState('Downloaded');
  const [carvedType, setCarvedType] = useState('Pcap');
  const [originUrl, setOriginUrl] = useState('');
  const [originName, setOriginName] = useState('');
  const [originTool, setOriginTool] = useState('');
  const [originParentFile, setOriginParentFile] = useState('');
  const [originToolFlags, setOriginToolFlags] = useState('');
  const [originSniffer, setOriginSniffer] = useState('');
  const [originSource, setOriginSource] = useState('');
  const [originDestination, setOriginDestination] = useState('');
  const [originIncident, setOriginIncident] = useState('');
  const [originCoverTerm, setOriginCoverTerm] = useState('');
  const [originMissionTeam, setOriginMissionTeam] = useState('');
  const [originNetwork, setOriginNetwork] = useState('');
  const [originMachine, setOriginMachine] = useState('');
  const [originLocation, setOriginLocation] = useState('');
  const [originMemoryType, setOriginMemoryType] = useState('');
  const [originReconstructed, setOriginReconstructed] = useState('');
  const [originBaseAddress, setOriginBaseAddress] = useState('');
  const [originSourceIp, setOriginSourceIp] = useState('');
  const [originDestinationIp, setOriginDestinationIp] = useState('');
  const [originSourcePort, setOriginSourcePort] = useState();
  const [originDestinationPort, setOriginDestinationPort] = useState();
  const [originProtocol, setOriginProtocol] = useState('');
  const [originCarvedPcapUrl, setOriginCarvedPcapUrl] = useState('');
  const [uploadError, setUploadError] = useState<string[]>([]);
  const [runReactionsRes, setRunReactionsRes] = useState('');
  const [uploadSHA256, setUploadSHA256] = useState<string[]>([]);
  const [tags, setTags] = useState([{ key: '', value: '' }]);
  const [selectedGroups, setSelectedGroups] = useState(entity ? entity.groups : []);
  const [reactionsList, setReactionsList] = useState([]);
  const [uploadInProgress, setUploadInProgress] = useState(false);
  const [activeUploads, setActiveUploads] = useState([]);
  const [uploadStatus, setUploadStatus] = useState({});
  const [uploadFailures, setUploadFailures] = useState({});
  const [uploadStatusDropdown, setUploadStatusDropdown] = useState({});
  const [uploadReactionRes, setUploadReactionRes] = useState([]); // Results from submissions
  const [uploadReactions, setUploadReactions] = useState({}); // All the reaction details
  const [uploadReactionFailures, setUploadReactionFailures] = useState(0);
  const [totalUploadSize, setTotalUploadSize] = useState(0);
  const [showUploadStatus, setShowUploadStatus] = useState(false);
  const [selectedTLP, setSelectedTLP] = useState({
    Red: false,
    White: false,
    Amber: false,
    Green: false,
  });
  const [controller, setController] = useState(new AbortController());

  // Make selected tlp tags
  const tlpTags = () => {
    const desiredTags = Object.keys(selectedTLP).filter((tlp) => {
      return selectedTLP[tlp];
    });
    const finalTlpTags = [];
    desiredTags.map((tlp) => {
      finalTlpTags.push({ key: 'TLP', value: tlp });
    });
    return finalTlpTags;
  };

  // Upload file(s) action
  const upload = async () => {
    setUploadStatus({});
    setUploadReactions({});
    setUploadReactionRes([]);
    setTotalUploadSize(0);
    setUploadReactionFailures(0);

    // check input and validate required fields
    const formBase = new FormData();
    if (filesArray.length == 0) {
      setUploadError(['Please select a file to upload']);
      return;
    }

    // Description
    if (description) {
      formBase.append('description', description);
    }
    // Groups
    if (selectedGroups.length > 0) {
      selectedGroups.map((group) => formBase.append('groups', group));
    } else {
      setUploadError(['At least one group must be selected to submit a file']);
      return;
    }

    // Add tlp tags to form if set
    const filteredTLPTags = tlpTags();
    if (filteredTLPTags) {
      filteredTLPTags.map((tag) => {
        // a valid key and value must be set for each uploaded tag
        if (tag['key'] && tag['value']) {
          formBase.append(`tags[${tag['key']}]`, tag['value']);
        }
      });
    }

    // Add tags to form if set
    if (tags) {
      // Tags
      tags.map((tag) => {
        // a valid key must be set for each uploaded tag
        if (tag['key'] && tag['value']) {
          formBase.append(`tags[${tag['key']}]`, tag['value']);
        }
      });
    }

    // Add downloaded origin info
    if (originType == 'Downloaded') {
      // make sure origin url is set
      if (originUrl) {
        // build origin form starting with type
        formBase.append('origin[origin_type]', originType);
        // add url
        formBase.append('origin[url]', originUrl);
        // see if optional site name is added
        if (originName) {
          formBase.append('origin[name]', originName);
        }
      } else if (originName) {
        setUploadError(['ORIGIN field "SITE NAME" set while necessary field "URL" is blank']);
        return;
      }
      // Add transformed origin info
    } else if (originType == 'Transformed') {
      if (originParentFile) {
        // build origin form starting with type
        formBase.append('origin[origin_type]', originType);
        formBase.append('origin[parent]', originParentFile);
        if (originTool) {
          formBase.append('origin[tool]', originTool);
        }
        if (originToolFlags) {
          formBase.append('origin[flags]', originToolFlags);
        }
      } else if (originTool) {
        setUploadError(['ORIGIN field "TOOL" set while necessary field "PARENT" is blank']);
        return;
      } else if (originToolFlags) {
        // Should this be extra-optional?
        // Meaning should it also depend on the tool being blank as well?
        setUploadError(['ORIGIN field "FLAGS" set while necessary field "PARENT" is blank']);
        return;
      }
      // Add unpacked origin info
    } else if (originType == 'Unpacked') {
      if (originParentFile) {
        // build origin form starting with type
        formBase.append('origin[origin_type]', originType);
        formBase.append('origin[parent]', originParentFile);
        if (originTool) {
          formBase.append('origin[tool]', originTool);
        }
        if (originToolFlags) {
          formBase.append('origin[flags]', originToolFlags);
        }
      } else if (originTool) {
        setUploadError(['ORIGIN field "TOOL" set while necessary field "PARENT" is blank']);
        return;
      } else if (originToolFlags) {
        // Should this be extra-optional?
        // Meaning should it also depend on the tool being blank as well?
        setUploadError(['ORIGIN field "FLAGS" set while necessary field "PARENT" is blank']);
        return;
      }
      // Add carved origin info
    } else if (originType == 'Carved') {
      if (originParentFile) {
        // build origin form starting with type
        if (!carvedType) {
          setUploadError(['ORIGIN "Carved" needs a specified type']);
          return;
        }
        formBase.append('origin[parent]', originParentFile);
        if (originTool) {
          formBase.append('origin[tool]', originTool);
        }
        let totalType = originType + carvedType;
        formBase.append('origin[origin_type]', totalType);
        if (totalType == 'CarvedPcap') {
          if (originSourceIp) {
            formBase.append('origin[src_ip]', originSourceIp);
          }
          if (originDestinationIp) {
            formBase.append('origin[dest_ip]', originDestinationIp);
          }
          if (originSourcePort) {
            formBase.append('origin[src_port]', originSourcePort);
          }
          if (originDestinationPort) {
            formBase.append('origin[dest_port]', originDestinationPort);
          }
          if (originProtocol) {
            formBase.append('origin[proto]', originProtocol);
          }
          if (originCarvedPcapUrl) {
            formBase.append('origin[url]', originCarvedPcapUrl);
          }
        }
      } else if (originTool) {
        setUploadError(['ORIGIN field "TOOL" set while necessary field "PARENT" is blank']);
        return;
      }
      // Add pulled off wire origin info
    } else if (originType == 'Wire') {
      if (originSniffer) {
        // build origin form starting with type
        formBase.append('origin[origin_type]', originType);
        formBase.append('origin[sniffer]', originSniffer);
        if (originSource) {
          formBase.append('origin[source]', originSource);
        }
        if (originDestination) {
          formBase.append('origin[destination]', originDestination);
        }
      } else if (originSource) {
        setUploadError(['ORIGIN field "SOURCE" set while necessary field "SNIFFER" is blank']);
        return;
      } else if (originDestination) {
        setUploadError(['ORIGIN field "DESTINATION" set while necessary field "SNIFFER" is blank']);
        return;
      }
      // Add incident origin info
    } else if (originType == 'Incident') {
      if (originIncident) {
        formBase.append('origin[origin_type]', originType);
        formBase.append('origin[incident]', originIncident);
        if (originMissionTeam) {
          formBase.append('origin[mission_team]', originMissionTeam);
        }
        if (originCoverTerm) {
          formBase.append('origin[cover_term]', originCoverTerm);
        }
        if (originNetwork) {
          formBase.append('origin[network]', originNetwork);
        }
        if (originMachine) {
          formBase.append('origin[machine]', originMachine);
        }
        if (originLocation) {
          formBase.append('origin[location]', originLocation);
        }
      } else if (originCoverTerm) {
        setUploadError(['ORIGIN field "COVER TERM" set while necessary field "INCIDENT ID" is blank']);
        return;
      } else if (originMissionTeam) {
        setUploadError(['ORIGIN field "MISSION TEAM" set while necessary field "INCIDENT ID" is blank']);
        return;
      } else if (originNetwork) {
        setUploadError(['ORIGIN field "NETWORK" set while necessary field "INCIDENT ID" is blank']);
        return;
      } else if (originMachine) {
        setUploadError(['ORIGIN field "MACHINE" set while necessary field "INCIDENT ID" is blank']);
        return;
      } else if (originLocation) {
        setUploadError(['ORIGIN field "LOCATION" set while necessary field "INCIDENT ID" is blank']);
        return;
      }
      // Add pulled memory dump origin info
    } else if (originType == 'MemoryDump') {
      if (originMemoryType) {
        // build origin form starting with type
        formBase.append('origin[origin_type]', originType);
        formBase.append('origin[memory_type]', originMemoryType);
        if (originParentFile) {
          formBase.append('origin[parent]', originParentFile);
        }
        if (originReconstructed) {
          formBase.append('origin[reconstructed]', originReconstructed);
        }
        if (originBaseAddress) {
          formBase.append('origin[base_addr]', originBaseAddress);
        }
      } else if (originParentFile) {
        setUploadError(['ORIGIN field "PARENT" set while necessary field "MEMORY TYPE" is blank']);
        return;
      } else if (originReconstructed.length > 0) {
        setUploadError(['ORIGIN field "RECONSTRUCTED" set while necessary field "MEMORY TYPE" is blank']);
        return;
      } else if (originBaseAddress) {
        setUploadError(['ORIGIN field "BASE ADDRESS" set while necessary field "MEMORY TYPE" is blank']);
        return;
      }
    }

    // We only want the more complex upload details to be shown if there is more than one upload
    // at a time. For a single file we can maintain something akin to the original format.
    if (filesArray.length > 1) {
      setShowUploadStatus(true);
    }

    const filesUploadProgress = {};
    const statusDropdown = {};
    let uploadSize = 0;
    setUploadInProgress(true);
    // loop though all files and submit file and jobs for each
    for (const submission of filesArray) {
      if (submission) {
        uploadSize = uploadSize + submission.size;
        filesUploadProgress[submission.path] = {
          progress: 0,
          size: submission.size,
          type: 'info',
          msg: 'Queued',
          sha256: '',
          fileFail: false,
          reactionFail: false,
        };
      }
      statusDropdown[submission.path] = false;
      setUploadReactions((uploadReactions) => ({
        ...uploadReactions,
        [submission.path]: [],
      }));
    }
    setTotalUploadSize(uploadSize);
    setUploadStatus(filesUploadProgress);
    setUploadStatusDropdown(statusDropdown);

    const uploadPromises = [];
    let currentUploadCount = 0;
    for (const submission of filesArray) {
      // We can't reuse the same form when submitting files async,
      // so copy the contents and create a new one for each.
      const newForm = new FormData();
      for (const [key, val] of formBase.entries()) {
        newForm.append(key, val);
      }
      if (submission) {
        newForm.set('data', submission);
      }

      // We have an artificial limit for parallel uploads, when we reach the limit
      // pause a moment before checking again. Without the pause, page will lock up.
      while (currentUploadCount >= PARALLELUPLOADLIMIT) {
        // Current wait is 1 second (1000 ms)
        await new Promise((f) => setTimeout(f, 1000));
      }
      currentUploadCount = currentUploadCount + 1;
      // Keep the promises so we can properly know when everything is done.
      uploadPromises.push(
        trackAndUploadFile(newForm).then(() => {
          currentUploadCount = currentUploadCount - 1;
        }),
      );
    }
    // Wait for all the promises to be done.
    Promise.all(uploadPromises).then((values) => {
      setUploadInProgress(false);
    });
  };

  // Calculate the overall completion percentage based on the total size of the files being
  // uploaded and the bytes already uploaded for each file. Returned as a percentage.
  const computeTotal = () => {
    let totalUploaded = 0;
    Object.values(uploadStatus).map((value, index) => {
      totalUploaded = totalUploaded + Math.ceil((value.progress / 100) * value.size);
    });
    const percentComplete = Math.floor((totalUploaded / totalUploadSize) * 100);
    return percentComplete;
  };

  // Base functionality for tracking the status of a file upload.
  const trackAndUploadFile = (form) => {
    const allResSha256 = [];
    const allResErrors = [];
    const submission = form.get('data');
    setActiveUploads((activeUploads) => [...activeUploads, submission.path]);

    // Callback functions for updating progress bar data as it arrives
    const uploadFileProgressHandler = (progress) => {
      let totalUploaded = 0;
      Object.values(uploadStatus).map((value, index) => {
        totalUploaded = totalUploaded + Math.ceil((value.progress / 100) * value.size);
      });
      // Only update progress here if it is less that 100% complete.
      // We fudge the value to 99% elsewhere and we don't want it going backwards.
      if (progress < 1) {
        setUploadStatus((uploadStatus) => ({
          ...uploadStatus,
          [submission.path]: {
            progress: Math.floor(progress * 100),
            size: submission.size,
            type: 'info',
            msg: 'Upload in progress',
            sha256: '',
            fileFail: false,
            reactionFail: false,
          },
        }));
      }
    };

    // Callback function for updating error values.
    const addUploadErrorMsg = (file, error) => {
      allResErrors.push(error);
      setUploadStatus((fileProgress) => ({
        ...fileProgress,
        [file.path]: {
          progress: 100,
          size: file.size,
          type: 'danger',
          msg: error,
          sha256: '',
          fileFail: true,
          reactionFail: true,
        },
      }));
    };

    // Upload the newly constructed form
    // Save the promise so we can wait on all the threads at the end indicating all the uploads
    // are complete.
    return uploadFile(form, (msg: string) => addUploadErrorMsg(submission, msg), uploadFileProgressHandler, controller).then(
      (response: any) => {
        if (response) {
          allResSha256.push(response.sha256);
          // Set progress to 99% while we submit reactions. Might be a better way to track this.
          setUploadStatus((fileProgress) => ({
            ...fileProgress,
            [submission.path]: {
              progress: 99,
              size: submission.size,
              type: 'info',
              msg: 'Submitting reactions',
              sha256: response.sha256,
              fileFail: false,
              reactionFail: false,
            },
          }));
          // Track reaction failure count. This is easier than counting the length
          // of another data structure. Assume they will fail before submitting them.
          // Also only doing this after file is uploaded properly since we don't
          // want to count them otherwise.
          setUploadReactionFailures((uploadReactionFailures) => uploadReactionFailures + reactionsList.length);
          trackAndSubmitReactions(response.sha256, submission, reactionsList);
          // Remove this submission from the set of failures if its in there
          if (uploadFailures.size > 0) {
            setUploadFailures((uploadFailures) => {
              // eslint-disable-next-line no-unused-vars
              const { [submission.path]: _, rest } = uploadFailures;
              return rest;
            });
          }
        } else {
          // Track files that were not submitted successfully. Keep the form for reuse.
          setUploadFailures((uploadFailures) => ({
            ...uploadFailures,
            [submission.path]: form,
          }));
        }
        setUploadSHA256([...allResSha256]);
        // If there are no errors, don't try and add anything to the state or
        // you will get weird undefined entries.
        if (allResErrors.length > 0) {
          setUploadError((uploadError) => [...uploadError, allResErrors]);
        }
        // Remove this submission from the set of currently in-progress uploads
        setActiveUploads((activeUploads) => activeUploads.filter((item) => item !== submission.path));
      },
    );
  };

  // Submit reactions and track their results.
  const trackAndSubmitReactions = (uploadSHA256: string, submission: any, submitReactionsList: any[]) => {
    const allRunReactionsRes = [];
    // Submit reactions from the list, return the promise so we can wait if needed
    // for multiple threads
    return submitReactions(uploadSHA256, submitReactionsList).then((submitRes) => {
      let error = false;
      // Check the status of the reactions and flag any errors.
      Object.values(submitRes).map((value, index) => {
        if (value.error != '') {
          error = true;
        } else {
          // Decrement failure count.
          setUploadReactionFailures((uploadReactionFailures) => uploadReactionFailures - 1);
        }
        const existingResult = uploadReactionRes.filter((result) => result.id === uploadSHA256 + value.pipeline);
        if (existingResult.length === 0) {
          // Slightly convoluted data structure. Using id that is a combo of the file SHA
          // and the specific pipeline being used. Almost certainly a better way to do this
          // but this certainly works for now.
          setUploadReactionRes((uploadReactionRes) => [
            ...uploadReactionRes,
            {
              id: uploadSHA256 + value.pipeline,
              sha256: uploadSHA256,
              result: value,
              submission: submission,
            },
          ]);
        } else {
          // This updates a reaction result state if it has changed.
          setUploadReactionRes((uploadReactionRes) =>
            uploadReactionRes.map((result) => {
              if (result.id === uploadSHA256 + value.pipeline) {
                return {
                  id: uploadSHA256 + value.pipeline,
                  sha256: uploadSHA256,
                  result: value,
                  submission: submission,
                };
              } else {
                return result;
              }
            }),
          );
        }
        submitRes[index] = {
          ...submitRes[index],
          path: submission.path,
          size: submission.size,
          sha256: uploadSHA256,
        };
      });
      // Store the submission result for general display according to the file they are
      // associated with.
      setUploadReactions((uploadReactions) => ({
        ...uploadReactions,
        [submission.path]: submitRes,
      }));
      allRunReactionsRes.push(...submitRes);
      // Call the file done after all reactions submitted. We will get here even if
      // there were no reactions to submit.
      if (error) {
        // error result
        setUploadStatus((fileProgress) => ({
          ...fileProgress,
          [submission.path]: {
            progress: 100,
            size: submission.size,
            type: 'warning',
            msg: 'Error submitting reactions',
            sha256: uploadSHA256,
            fileFail: false,
            reactionFail: true,
          },
        }));
      } else {
        // Successful result
        setUploadStatus((fileProgress) => ({
          ...fileProgress,
          [submission.path]: {
            progress: 100,
            size: submission.size,
            type: 'success',
            msg: 'Upload successful!',
            sha256: uploadSHA256,
            fileFail: false,
            reactionFail: false,
          },
        }));
      }
      setRunReactionsRes([...allRunReactionsRes]);
    });
  };

  // Retry a single file upload that has previously failed
  const retryFileUpload = (fileName) => {
    setUploadInProgress(true);
    trackAndUploadFile(uploadFailures[fileName]).then(() => setUploadInProgress(false));
  };

  // Retry all file uploads that have previously failed
  const retryAllFileUploads = async () => {
    setUploadInProgress(true);
    setUploadFailures(() => ({}));
    const uploadPromises = [];
    let currentUploadCount = 0;
    Object.entries(uploadFailures).map(async ([key, form]) => {
      // We have an artificial limit for parallel uploads, when we reach the limit
      // pause a moment before checking again. Without the pause, page will lock up.
      while (currentUploadCount >= PARALLELUPLOADLIMIT) {
        // Current wait is 1 second (1000 ms)
        await new Promise((f) => setTimeout(f, 1000));
      }
      currentUploadCount = currentUploadCount + 1;
      uploadPromises.push(
        trackAndUploadFile(form).then(() => {
          currentUploadCount = currentUploadCount - 1;
        }),
      );
    });
    Promise.all(uploadPromises).then((values) => {
      setUploadInProgress(false);
    });
  };

  // Retry submitting a single reaction for a single file.
  const retrySubmitReaction = (status) => {
    setUploadInProgress(true);
    // Set progress to 99% while we submit reactions. Might be a better way to track this.
    setUploadStatus((fileProgress) => ({
      ...fileProgress,
      [status.submission.path]: {
        progress: 99,
        size: status.submission.size,
        type: 'info',
        msg: 'Submitting reactions',
        sha256: status.sha256,
        fileFail: false,
        reactionFail: false,
      },
    }));
    // Get the specific details for the reaction including the form
    // that was originally used.
    const failedReaction = uploadReactionRes.filter((failure) => failure.id == status.id)[0];
    trackAndSubmitReactions(
      status.sha256,
      failedReaction.submission,
      reactionsList.filter((reaction) => reaction.pipeline === failedReaction.result.pipeline),
    ).then(() => setUploadInProgress(false));
  };

  // Retry all of the reactions that have failed.
  const retryAllReactionSubmissions = () => {
    setUploadInProgress(true);
    const submissionPromises = [];
    // Iterate through the reactions and look for ones with an error.
    uploadReactionRes.map((value) => {
      if (value.result.error !== '') {
        // Get the specific details for the reaction including the form
        // that was originally used.
        const failedReaction = uploadReactionRes.filter((failure) => failure.id == value.id)[0];
        submissionPromises.push(
          trackAndSubmitReactions(
            value.sha256,
            failedReaction.submission,
            reactionsList.filter((reaction) => reaction.pipeline === failedReaction.result.pipeline),
          ),
        );
      }
    });
    Promise.all(submissionPromises).then((values) => {
      setUploadInProgress(false);
    });
  };

  // Cancel an upload in progress
  const cancelUpload = () => {
    controller.abort();
    // Reset controller so everything can be tried again.
    setController(new AbortController());
  };

  // Set message states back to default.
  const resetStatusMessages = () => {
    setUploadSHA256([]);
    setUploadError([]);
    setRunReactionsRes([]);
    setUploadStatus({});
    setUploadReactions({});
    setUploadReactionRes([]);
    setUploadReactionFailures(0);
  };

  const AlertBanner = () => {
    return (
      <Fragment>
        <Row>
          {uploadSHA256 &&
            uploadSHA256.map((sha256) => (
              <Alert className="d-flex justify-content-center" key={sha256} variant="success">
                File uploaded successfully: <pre> </pre>
                <Link className="link-text" to={'/file/' + sha256} target="_blank">
                  {sha256}
                </Link>
              </Alert>
            ))}
        </Row>
        <Row>
          {uploadError &&
            uploadError.map((message) => (
              <Alert className="d-flex justify-content-center" key={message} variant="danger">
                {message}
              </Alert>
            ))}
        </Row>
        <RunReactionAlerts responses={runReactionsRes} />
      </Fragment>
    );
  };

  // Allows user to select TLP tags for file from buttons: red,white,amber,green
  const TLPSelection = () => {
    return (
      <Card className="panel">
        <Card.Body className="d-flex justify-content-center">
          {Object.keys(TLPColors).map((tlp) => (
            <Button
              variant=""
              className={`tlp-btn ${TLPColors[tlp]}-btn ${selectedTLP[tlp] ? 'selected' : ''}`}
              key={tlp}
              onClick={(e) => {
                const tempSelection = {};
                Object.keys(TLPColors).map((color) => {
                  if (color != tlp) {
                    tempSelection[color] = false;
                  } else {
                    if (selectedTLP[color] == true) {
                      tempSelection[color] = false;
                    } else {
                      tempSelection[color] = true;
                    }
                  }
                });
                setSelectedTLP(tempSelection);
                resetStatusMessages();
              }}
            >
              <b>{tlp}</b>
            </Button>
          ))}
        </Card.Body>
      </Card>
    );
  };

  const selectOrigin = () => {
    return (
      <Card className="panel">
        <Card.Body>
          <Tabs fill activeKey={originType} onSelect={(k) => setOriginType(k)}>
            <Tab eventKey="Downloaded" title="Downloaded">
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>URL</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    value={originUrl}
                    placeholder="badsite.xyz"
                    onChange={(e) => {
                      setOriginUrl(String(e.target.value));
                      resetStatusMessages();
                    }}
                    isInvalid={!originUrl && originName}
                  />
                  <Form.Control.Feedback type="invalid">Please enter a site URL.</Form.Control.Feedback>
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Site Name</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    value={originName}
                    placeholder="optional"
                    onChange={(e) => {
                      setOriginName(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
            </Tab>
            <Tab eventKey="Transformed" title="Transformed">
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Parent</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    placeholder="SHA256"
                    value={originParentFile}
                    onChange={(e) => {
                      setOriginParentFile(String(e.target.value));
                      resetStatusMessages();
                    }}
                    isInvalid={!originParentFile && (originTool || originToolFlags)}
                  />
                  <Form.Control.Feedback type="invalid">Please enter a SHA256 value for the Parent.</Form.Control.Feedback>
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Tool</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    placeholder="optional"
                    value={originTool}
                    onChange={(e) => {
                      setOriginTool(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Flags</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    value={originToolFlags}
                    placeholder="optional"
                    onChange={(e) => {
                      setOriginToolFlags(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
            </Tab>
            <Tab eventKey="Unpacked" title="Unpacked">
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Parent</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    placeholder="SHA256"
                    value={originParentFile}
                    onChange={(e) => {
                      setOriginParentFile(String(e.target.value));
                      resetStatusMessages();
                    }}
                    isInvalid={!originParentFile && (originTool || originToolFlags)}
                  />
                  <Form.Control.Feedback type="invalid">Please enter a SHA256 value for the Parent.</Form.Control.Feedback>
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Tool</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    placeholder="optional"
                    value={originTool}
                    onChange={(e) => {
                      setOriginTool(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Flags</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    value={originToolFlags}
                    placeholder="optional"
                    onChange={(e) => {
                      setOriginToolFlags(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
            </Tab>
            <Tab eventKey="Carved" title="Carved">
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Parent</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    placeholder="SHA256"
                    value={originParentFile}
                    onChange={(e) => {
                      setOriginParentFile(String(e.target.value));
                      resetStatusMessages();
                    }}
                    isInvalid={!originParentFile && (originTool || originToolFlags)}
                  />
                  <Form.Control.Feedback type="invalid">Please enter a SHA256 value for the Parent.</Form.Control.Feedback>
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Tool</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    placeholder="optional"
                    value={originTool}
                    onChange={(e) => {
                      setOriginTool(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
              <Row>
                <Card className="panel" style={{ boxShadow: 0, border: 'none' }}>
                  <Card.Body>
                    <Tabs fill activeKey={carvedType} onSelect={(k) => setCarvedType(k)}>
                      <Tab eventKey="Pcap" title="PCAP">
                        <br />
                        <Row>
                          <Col className="name-width" xs={2}>
                            <Subtitle>Source IP</Subtitle>
                          </Col>
                          <Col xs={5}>
                            <Form.Control
                              type="text"
                              placeholder="optional"
                              value={originSourceIp}
                              onChange={(e) => {
                                setOriginSourceIp(String(e.target.value));
                                resetStatusMessages();
                              }}
                              isInvalid={originSourceIp && !isIP(originSourceIp)}
                            />
                            <Form.Control.Feedback type="invalid">Please enter a valid IPv4/IPv6 address.</Form.Control.Feedback>
                          </Col>
                        </Row>
                        <br />
                        <Row>
                          <Col className="name-width" xs={2}>
                            <Subtitle>Destination IP</Subtitle>
                          </Col>
                          <Col xs={5}>
                            <Form.Control
                              type="text"
                              placeholder="optional"
                              value={originDestinationIp}
                              onChange={(e) => {
                                setOriginDestinationIp(String(e.target.value));
                                resetStatusMessages();
                              }}
                              isInvalid={originDestinationIp && !isIP(originDestinationIp)}
                            />
                            <Form.Control.Feedback type="invalid">Please enter a valid IPv4/IPv6 address.</Form.Control.Feedback>
                          </Col>
                        </Row>
                        <br />
                        <Row>
                          <Col className="name-width" xs={2}>
                            <Subtitle>Source Port</Subtitle>
                          </Col>
                          <Col xs={5}>
                            <Form.Control
                              type="number"
                              placeholder="optional"
                              value={originSourcePort}
                              onChange={(e) => {
                                setOriginSourcePort(e.target.value);
                                resetStatusMessages();
                              }}
                              isInvalid={originSourcePort && (originSourcePort < 1 || originSourcePort > 65535)}
                            />
                            <Form.Control.Feedback type="invalid">
                              Please enter a valid port (between 1 and 65535, inclusive).
                            </Form.Control.Feedback>
                          </Col>
                        </Row>
                        <br />
                        <Row>
                          <Col className="name-width" xs={2}>
                            <Subtitle>Destination Port</Subtitle>
                          </Col>
                          <Col xs={5}>
                            <Form.Control
                              type="number"
                              placeholder="optional"
                              value={originDestinationPort}
                              onChange={(e) => {
                                setOriginDestinationPort(e.target.value);
                                resetStatusMessages();
                              }}
                              isInvalid={originDestinationPort && (originDestinationPort < 1 || originDestinationPort > 65535)}
                            />
                            <Form.Control.Feedback type="invalid">
                              Please enter a valid port (between 1 and 65535, inclusive).
                            </Form.Control.Feedback>
                          </Col>
                        </Row>
                        <br />
                        <Row>
                          <Col className="name-width" xs={2}>
                            <Subtitle>Protocol</Subtitle>
                          </Col>
                          <Col xs={5}>
                            <Form.Control
                              type="text"
                              placeholder="TCP/UDP (optional)"
                              value={originProtocol}
                              onChange={(e) => {
                                setOriginProtocol(e.target.value);
                                resetStatusMessages();
                              }}
                              isInvalid={
                                originProtocol &&
                                originProtocol != 'TCP' &&
                                originProtocol != 'Tcp' &&
                                originProtocol != 'tcp' &&
                                originProtocol != 'UDP' &&
                                originProtocol != 'Udp' &&
                                originProtocol != 'udp'
                              }
                            />
                            <Form.Control.Feedback type="invalid">
                              Please enter a valid protocol ("TCP/Tcp/tcp" or "UDP/Udp/udp").
                            </Form.Control.Feedback>
                          </Col>
                        </Row>
                        <br />
                        <Row>
                          <Col className="name-width" xs={2}>
                            <Subtitle>URL</Subtitle>
                          </Col>
                          <Col xs={5}>
                            <Form.Control
                              type="text"
                              value={originCarvedPcapUrl}
                              placeholder="optional"
                              onChange={(e) => {
                                setOriginCarvedPcapUrl(String(e.target.value));
                                resetStatusMessages();
                              }}
                            />
                          </Col>
                        </Row>
                      </Tab>
                      <Tab eventKey="Unknown" title="Unknown"></Tab>
                    </Tabs>
                  </Card.Body>
                </Card>
              </Row>
            </Tab>
            <Tab eventKey="Wire" title="Wire">
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Sniffer</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    placeholder="name"
                    value={originSniffer}
                    onChange={(e) => {
                      setOriginSniffer(String(e.target.value));
                      resetStatusMessages();
                    }}
                    isInvalid={!originSniffer && (originSource || originDestination)}
                  />
                  <Form.Control.Feedback type="invalid">Please enter a name for the Sniffer.</Form.Control.Feedback>
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Source</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    placeholder="optional"
                    value={originSource}
                    onChange={(e) => {
                      setOriginSource(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Destination</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    placeholder="optional"
                    value={originDestination}
                    onChange={(e) => {
                      setOriginDestination(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
            </Tab>
            <Tab eventKey="Incident" title="Incident">
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Incident ID</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    value={originIncident}
                    placeholder="name"
                    onChange={(e) => {
                      setOriginIncident(String(e.target.value));
                      resetStatusMessages();
                    }}
                    isInvalid={
                      !originIncident && (originCoverTerm || originMissionTeam || originNetwork || originMachine || originLocation)
                    }
                  />
                  <Form.Control.Feedback type="invalid">Please enter an Incident ID.</Form.Control.Feedback>
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Cover Term</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    value={originCoverTerm}
                    placeholder="optional"
                    onChange={(e) => {
                      setOriginCoverTerm(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Mission Team</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    value={originMissionTeam}
                    placeholder="optional"
                    onChange={(e) => {
                      setOriginMissionTeam(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Network</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    placeholder="optional"
                    value={originNetwork}
                    onChange={(e) => {
                      setOriginNetwork(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Machine</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    placeholder="optional"
                    value={originMachine}
                    onChange={(e) => {
                      setOriginMachine(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Location</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    placeholder="optional"
                    value={originLocation}
                    onChange={(e) => {
                      setOriginLocation(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
            </Tab>
            <Tab eventKey="MemoryDump" title="Memory Dump">
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Memory Type</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    value={originMemoryType}
                    placeholder="type"
                    onChange={(e) => {
                      setOriginMemoryType(String(e.target.value));
                      resetStatusMessages();
                    }}
                    isInvalid={!originMemoryType && (originParentFile || originReconstructed.length > 0 || originBaseAddress)}
                  />
                  <Form.Control.Feedback type="invalid">Please enter a Memory Type.</Form.Control.Feedback>
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Parent</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    placeholder="optional"
                    value={originParentFile}
                    onChange={(e) => {
                      setOriginParentFile(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Reconstructed</Subtitle>
                </Col>
                <Col xs={5}>
                  <SelectableArray initialEntries={[]} setEntries={setOriginReconstructed} disabled={false} placeholder="optional" />
                </Col>
              </Row>
              <br />
              <Row>
                <Col className="name-width" xs={2}>
                  <Subtitle>Base Address</Subtitle>
                </Col>
                <Col xs={5}>
                  <Form.Control
                    type="text"
                    value={originBaseAddress}
                    placeholder="optional"
                    onChange={(e) => {
                      setOriginBaseAddress(String(e.target.value));
                      resetStatusMessages();
                    }}
                  />
                </Col>
              </Row>
            </Tab>
          </Tabs>
        </Card.Body>
      </Card>
    );
  };

  return (
    <>
      {showUploadStatus && (
        <Fragment>
          Total
          <Row className="upload-bar">
            <Col>
              <ProgressBarContainer name={'Total'} value={computeTotal()} error={uploadError.length} />
            </Col>
          </Row>
          {uploadInProgress && (
            <Row className="upload-bar">
              {Object.values(activeUploads).map((key) => (
                <OverlayTipTop key={key} tip={uploadStatus[key].msg}>
                  {key}
                  <ProgressBarContainer name={key} value={uploadStatus[key].progress} error={uploadStatus[key].length} />
                </OverlayTipTop>
              ))}
            </Row>
          )}
          {!uploadInProgress && (
            <Card className="stats-container panel">
              <Card.Body>
                <div>{Object.keys(uploadStatus).length - Object.keys(uploadFailures).length} Files Uploaded Successfully</div>
                {Object.keys(uploadFailures).length > 0 && (
                  <div>
                    {Object.keys(uploadFailures).length} File Upload Failure(s)
                    <Button
                      size="xsm"
                      variant="no-outline-secondary"
                      className="retry-button"
                      onClick={() => {
                        retryAllFileUploads();
                      }}
                    >
                      {' '}
                      <FaRedo />
                    </Button>
                  </div>
                )}
                <div>{uploadReactionRes.length - uploadReactionFailures} Reaction(s) Submitted Successfully</div>
                {uploadReactionFailures > 0 && (
                  <div>
                    {uploadReactionFailures} Reaction Submission(s) Failed
                    <Button
                      size="xsm"
                      variant="no-outline-secondary"
                      className="retry-button"
                      onClick={() => {
                        retryAllReactionSubmissions();
                      }}
                    >
                      {' '}
                      <FaRedo />
                    </Button>
                  </div>
                )}
              </Card.Body>
            </Card>
          )}
          <Row className="mt-1">
            <Card className="panel">
              <Row>
                <Col className="status-dropdown" md={1} />
                <Col className="status-file" md={1}>
                  <Subtitle>Filename</Subtitle>
                </Col>
                <Col className="status-msg" md={1}>
                  <Subtitle>Status</Subtitle>
                </Col>
                <Col className="status-percent" md={1}>
                  <Subtitle>Progress</Subtitle>
                </Col>
                <Col className="status-sha-head">
                  <Subtitle>SHA256</Subtitle>
                </Col>
              </Row>
            </Card>
          </Row>
          <Row className="mt-1">
            {Object.entries(uploadStatus).map(([key, value]) => (
              <Fragment key={key}>
                <Card className="highlight-card">
                  <Row>
                    <Col className="status-dropdown" md={1}>
                      <Button
                        size="xsm"
                        variant="no-outline-secondary"
                        onClick={() =>
                          setUploadStatusDropdown((uploadStatusDropdown) => ({
                            ...uploadStatusDropdown,
                            [key]: !uploadStatusDropdown[key],
                          }))
                        }
                      >
                        {uploadStatusDropdown[key] ? <FaChevronUp /> : <FaChevronDown />}
                      </Button>
                    </Col>
                    <Col className="status-file" md={1}>
                      {key}
                    </Col>
                    <Col className={'status-msg' + (value.fileFail | value.reactionFail ? ' status-error' : '')} md={1}>
                      {value.msg}
                    </Col>
                    <Col className="status-percent" md={1}>
                      {value.fileFail ? '0' : value.progress}%
                    </Col>
                    {value.sha256 && (
                      <Link to={`/file/${value.sha256}`} target="_blank" className="status-sha-link link-text-alt">
                        <Col className="status-sha">{value.sha256}</Col>
                      </Link>
                    )}
                    {!value.sha256 && !uploadInProgress && (
                      <Col className="status-sha">
                        <Button size="xsm" variant="no-outline-secondary" className="redo-btn" onClick={(e) => retryFileUpload(key)}>
                          <FaRedo />
                        </Button>
                      </Col>
                    )}
                  </Row>
                </Card>
                {uploadStatusDropdown[key] &&
                  uploadReactions[key] &&
                  (uploadReactions[key].length === 0 ? (
                    <Row className="upload-content">
                      <b>No Reactions Submitted</b>
                    </Row>
                  ) : (
                    <div className="reaction-uploads-card">
                      <Card className="panel">
                        <Row className="reaction-row mt-1">
                          <Col md={2}>
                            <Subtitle>Pipeline</Subtitle>
                          </Col>
                          <Col md={2}>
                            <Subtitle>Group</Subtitle>
                          </Col>
                          <Col md={3}>
                            <Subtitle>Error</Subtitle>
                          </Col>
                          <Col md={2}>
                            <Subtitle>ID</Subtitle>
                          </Col>
                        </Row>
                      </Card>
                      {/* eslint-disable-next-line max-len*/}
                      {uploadReactionRes
                        .filter((result) => result.sha256 === value.sha256)
                        .map((val) => (
                          <Card className="highlight-card panel" key={val.result.id}>
                            <Row className="reaction-row">
                              <Col md={2}>{val.result.pipeline}</Col>
                              <Col md={2}>{val.result.group}</Col>
                              <Col className={val.result.error ? 'status-error' : ''} md={3}>
                                {val.result.error && val.result.error.split(':')[1]}
                              </Col>
                              <Col>
                                {val.result.error && !uploadInProgress && (
                                  <Button
                                    size="xsm"
                                    variant="no-outline-secondary"
                                    className="redo-btn"
                                    onClick={() => retrySubmitReaction(val)}
                                  >
                                    <FaRedo />
                                  </Button>
                                )}
                                <Link target="_blank" className="link-text-alt" to={`/reaction/${val.result.group}/${val.result.id}`}>
                                  {val.result.id}
                                </Link>
                              </Col>
                            </Row>
                          </Card>
                        ))}
                    </div>
                  ))}
              </Fragment>
            ))}
          </Row>
          {!uploadInProgress ? (
            <Col className="d-flex justify-content-center close-button">
              <Button
                className="ok-btn"
                onClick={(e) => {
                  resetStatusMessages();
                  setShowUploadStatus(false);
                }}
              >
                Back
              </Button>
            </Col>
          ) : (
            <Col className="d-flex justify-content-center close-button">
              <Button
                className="warning-btn"
                onClick={() => {
                  cancelUpload();
                }}
              >
                Cancel
              </Button>
            </Col>
          )}
        </Fragment>
      )}
      {!showUploadStatus && (
        <>
          <Row className="mb-4 alt-label">
            <Col className="upload-field-name"></Col>
            <Col className="upload-field-name-alt">
              <Subtitle>
                File <sup>*</sup>
              </Subtitle>
            </Col>
          </Row>
          <Row>
            <Col className="upload-field-name">
              <Subtitle>
                File <sup>*</sup>
              </Subtitle>
            </Col>
            <Col className={(uploadInProgress ? 'disabled ' : '') + 'upload-field'}>
              <UploadDropzone onChange={setFilesArray} setError={setUploadError} selectedFiles={filesArray} />
              <br />
            </Col>
          </Row>
          <Row className="mb-4 alt-label">
            <Col className="upload-field-name"></Col>
            <Col className="upload-field-name-alt">
              <Subtitle>
                Groups <sup>*</sup>
              </Subtitle>
            </Col>
          </Row>
          <Row className="mb-4">
            <Col className="upload-field-name">
              <Subtitle>
                Groups <sup>*</sup>
              </Subtitle>
            </Col>
            <Col className={(uploadInProgress ? 'disabled ' : '') + 'upload-field'}>
              <SelectInputArray
                isCreatable={false}
                options={userInfo?.groups ? userInfo.groups : []}
                values={selectedGroups}
                onChange={(groups: string[]) => setSelectedGroups(groups)}
              />
            </Col>
          </Row>
          <Row className="mb-4 alt-label">
            <Col className="upload-field-name"></Col>
            <Col className="upload-field-name-alt">
              <Subtitle>Description</Subtitle>
            </Col>
          </Row>
          <Row>
            <Col className="upload-field-name">
              <Subtitle>Description</Subtitle>
            </Col>
            <Col className={(uploadInProgress ? 'disabled ' : '') + 'upload-field'}>
              <Form.Control
                className="description-field"
                as="textarea"
                placeholder="Add Description"
                value={description}
                onChange={(e) => {
                  setDescription(e.target.value);
                  resetStatusMessages();
                }}
              />
            </Col>
          </Row>
          <Row className="mb-4 alt-label">
            <Col className="upload-field-name"></Col>
            <Col className="upload-field-name-alt">
              <Subtitle>Tags</Subtitle>
            </Col>
          </Row>
          <Row className="mb-2">
            <Col className="upload-field-name">
              <Subtitle>Tags</Subtitle>
            </Col>
            <Col className={(uploadInProgress ? 'disabled ' : '') + 'upload-field'}>
              <SelectableDictionary
                entries={tags}
                setEntries={setTags}
                keyPlaceholder={'Add Tag Key'}
                valuePlaceholder={'Add Tag Value'}
                setError={setUploadError}
              />
            </Col>
          </Row>
          <Row className="mb-4 alt-label">
            <Col className="upload-field-name"></Col>
            <Col className="upload-field-name-alt">
              <Subtitle>
                TLP <sup>T</sup>
              </Subtitle>
            </Col>
          </Row>
          <Row className="mb-2">
            <Col className="upload-field-name">
              <Subtitle>
                TLP <sup>T</sup>
              </Subtitle>
            </Col>
            <Col className={(uploadInProgress ? 'disabled ' : '') + 'upload-field'}>
              <TLPSelection />
            </Col>
          </Row>
          <Row className="mb-4 alt-label">
            <Col className="upload-field-name"></Col>
            <Col className="upload-field-name-alt">
              <Subtitle>
                Origin <sup>T</sup>
              </Subtitle>
            </Col>
          </Row>
          <Row className="mb-4">
            <Col className="upload-field-name">
              <Subtitle>
                Origin <sup>T</sup>
              </Subtitle>
            </Col>
            <Col className={(uploadInProgress ? 'disabled ' : '') + 'upload-field'}>{selectOrigin()}</Col>
          </Row>
          <Row className="mb-4 alt-label">
            <Col className="upload-field-name"></Col>
            <Col className="upload-field-name-alt">
              <Subtitle>Run Pipelines</Subtitle>
            </Col>
          </Row>
          <Row>
            <Col className="upload-field-name">
              <Subtitle>Run Pipelines</Subtitle>
            </Col>
            <Col className={(uploadInProgress ? 'disabled ' : '') + 'upload-field'}>
              <SelectPipelines
                userInfo={userInfo}
                setReactionsList={setReactionsList}
                setError={setUploadError}
                currentSelections={reactionsList}
              />
            </Col>
          </Row>
          <Row className="mt-3">
            <Col className="upload-field-name" />
            <Col className="upload-field ms-4">
              <p>
                <sup>*</sup> This field is required.
              </p>
            </Col>
          </Row>
          <Row>
            <Col className="upload-field-name" />
            <Col className="upload-field ms-4">
              <p>
                <sup>T</sup> This field also creates tags when specified.
              </p>
            </Col>
          </Row>
          <Row className="d-flex justify-content-center">
            <Col className="upload-field-name"></Col>
            <Col className="upload-field">
              {uploadStatus && Object.entries(uploadStatus).length > 0 && (
                <Row className="upload-bar mt-3">
                  {Object.entries(uploadStatus).map(([key, value]) => (
                    <OverlayTipTop key={key} tip={value.msg}>
                      {key}
                      <ProgressBarContainer name={key} value={value.progress} error={uploadError.length} />
                    </OverlayTipTop>
                  ))}
                </Row>
              )}
              {!uploadInProgress && (
                <>
                  <Row className="upload_alerts">
                    <Col className="upload-field">
                      <AlertBanner />
                    </Col>
                  </Row>
                  <Row className="d-flex justify-content-center upload-btn">
                    <Col className="upload-field">
                      <center>
                        <Button className="ok-btn" onClick={() => upload()}>
                          Upload
                        </Button>
                      </center>
                    </Col>
                  </Row>
                </>
              )}
            </Col>
          </Row>
        </>
      )}
    </>
  );
};
