import { useEffect, useState } from 'react';
import { Alert, Button, Card, Col, Row } from 'react-bootstrap';
import { default as MarkdownHtml } from 'react-markdown';
import remarkGfm from 'remark-gfm';

// project imports
import { orderComparePipelineName } from './pipelines';
import Title from '@components/shared/titles/Title';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import { listPipelines } from '@thorpi/pipelines';
import { buildReactionsList } from './reactions';

// Component for allowing a user to select pipelines to run on a given sha256
const SelectPipelines = ({ userInfo, setReactionsList, setError, currentSelections }) => {
  const [pipelines, setPipelines] = useState({});
  const [selectedPipelines, setSelectedPipelines] = useState({});
  const [pipelinesListErrors, setPipelinesListErrors] = useState([]);
  // Get detailed pipelines info and a list of groups names
  useEffect(() => {
    let isSubscribed = true;
    const fetchData = async () => {
      const allPipelines = [];
      // dictionaries of name: boolean pairs representing whether a user has
      // selected a given pipeline to run or group to give access to a set
      // of uploaded files
      const selectablePipelines = {};
      const errors = [];
      if (userInfo && userInfo.groups) {
        for (const group of userInfo.groups) {
          const groupPipelines = await listPipelines(group, (error) => errors.push(error), true);
          if (groupPipelines) {
            allPipelines[group] = [...groupPipelines];
            groupPipelines.map((pipeline) => {
              // selectablePipelines =
              //   {group1: {pipeline1: false, pipeline2: false}, group2: {pipeline3: false}}
              const tempSelected = {};
              tempSelected[`${pipeline.name}`] = false;
              if (pipeline.name in selectablePipelines) {
                selectablePipelines[`${pipeline.group}`][`${pipeline.name}`] = false;
              } else {
                selectablePipelines[`${pipeline.group}`] = tempSelected;
              }
            });
          }
        }
      }
      if (currentSelections) {
        currentSelections.forEach((pipeline) => (selectablePipelines[`${pipeline.pipeline}_${pipeline.group}`] = true));
      }
      // set local errors from listing pipelines in each group
      setPipelinesListErrors(errors);
      // don't update if resource is no longer mounted
      if (isSubscribed) {
        // save detailed pipeline info and map of pipeline names
        setPipelines(allPipelines);
        setSelectedPipelines(selectablePipelines);
      }
    };
    fetchData();
    return () => {
      isSubscribed = false;
    };
  }, [userInfo]);

  return (
    <Card className="panel">
      <Card.Body className="py-0">
        <center>
          {pipelinesListErrors &&
            pipelinesListErrors.map((error, idx) => (
              <Alert key={`pipeline-list-error-${idx}`} variant="" className="d-flex justify-content-center danger">
                {error}
              </Alert>
            ))}
          {pipelines &&
            userInfo &&
            userInfo['groups'] &&
            userInfo.groups.map((group) => {
              if (pipelines[group] && pipelines[group].length) {
                return (
                  <div key={group}>
                    <Row className="mt-4 mb-2">
                      <Col>
                        <Title small>{group}</Title>
                      </Col>
                    </Row>
                    <Row className="mb-4">
                      <Col>
                        {pipelines[group]
                          .sort((a, b) => orderComparePipelineName(a, b))
                          .map((pipeline) =>
                            pipeline.description != null ? (
                              <OverlayTipTop
                                key={`${pipeline.group}_${pipeline.name}`}
                                wide
                                tip={<MarkdownHtml remarkPlugins={[remarkGfm]}>{pipeline.description}</MarkdownHtml>}
                              >
                                <Button
                                  variant=""
                                  // eslint-disable-next-line max-len
                                  className={`m-1 primary-btn ${selectedPipelines[`${pipeline.group}`][`${pipeline.name}`] ? 'selected' : 'unselected'}`}
                                  onClick={(e) => {
                                    const selected = structuredClone(selectedPipelines);
                                    selected[`${pipeline.group}`][`${pipeline.name}`] =
                                      !selectedPipelines[`${pipeline.group}`][`${pipeline.name}`];
                                    setSelectedPipelines(selected);
                                    setReactionsList(buildReactionsList(selected));
                                    if (setError) setError([]);
                                  }}
                                >
                                  <font size="3">
                                    <b>{pipeline.name}</b>
                                  </font>
                                </Button>
                              </OverlayTipTop>
                            ) : (
                              <Button
                                variant=""
                                key={`${pipeline.group}_${pipeline.name}`}
                                // eslint-disable-next-line max-len
                                className={`m-1 primary-btn ${selectedPipelines[`${pipeline.group}`][`${pipeline.name}`] ? 'selected' : 'unselected'}`}
                                onClick={(e) => {
                                  const selected = structuredClone(selectedPipelines);
                                  selected[`${pipeline.group}`][`${pipeline.name}`] =
                                    !selectedPipelines[`${pipeline.group}`][`${pipeline.name}`];
                                  setSelectedPipelines(selected);
                                  setReactionsList(buildReactionsList(selected));
                                  if (setError) setError([]);
                                }}
                              >
                                <font size="3">
                                  <b>{pipeline.name}</b>
                                </font>
                              </Button>
                            ),
                          )}
                      </Col>
                    </Row>
                  </div>
                );
              } else {
                return null;
              }
            })}
        </center>
      </Card.Body>
    </Card>
  );
};

export default SelectPipelines;
