import { useEffect, useState } from 'react';
import { Button, Card, Col, Row } from 'react-bootstrap';
import AlertBanner from '@components/shared/alerts/AlertBanner';

// project imports
import { orderComparePipelineName } from './pipelines';
import { buildReactionsList, SelectedPipelines } from './reactions';
import Title from '@components/shared/titles/Title';
import Markdown from '@components/shared/syntax/Markdown';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import { listPipelines } from '@thorpi/pipelines';
import type { Pipeline } from '@models/pipelines';
import type { ReactionSelection } from '@models/reactions';
import type { UserInfo } from '@models/users';

interface SelectPipelinesProps {
  userInfo: UserInfo | null;
  setReactionsList: (list: ReactionSelection[]) => void;
  setError?: (errors: string[]) => void;
  currentSelections?: ReactionSelection[];
  sha256?: string;
}

const SelectPipelines = ({ userInfo, setReactionsList, setError, currentSelections }: SelectPipelinesProps) => {
  const [pipelines, setPipelines] = useState<Record<string, Pipeline[]>>({});
  const [selectedPipelines, setSelectedPipelines] = useState<SelectedPipelines>({});
  const [pipelinesListErrors, setPipelinesListErrors] = useState<string[]>([]);

  // get detailed pipelines info
  useEffect(() => {
    let isSubscribed = true;
    const fetchData = async () => {
      const allPipelines: Record<string, Pipeline[]> = {};
      const selectablePipelines: SelectedPipelines = {};
      const errors: string[] = [];
      if (userInfo && userInfo.groups) {
        for (const group of userInfo.groups) {
          const groupPipelines = (await listPipelines(group, (error: string) => errors.push(error), true)) as Pipeline[] | null;
          if (groupPipelines) {
            allPipelines[group] = [...groupPipelines];
            groupPipelines.forEach((pipeline: Pipeline) => {
              if (pipeline.group in selectablePipelines) {
                selectablePipelines[pipeline.group][pipeline.name] = false;
              } else {
                selectablePipelines[pipeline.group] = { [pipeline.name]: false };
              }
            });
          }
        }
      }
      if (currentSelections) {
        currentSelections.forEach((selection) => {
          if (!(selection.group in selectablePipelines)) {
            selectablePipelines[selection.group] = {};
          }
          selectablePipelines[selection.group][selection.pipeline] = true;
        });
      }
      setPipelinesListErrors(errors);
      if (isSubscribed) {
        setPipelines(allPipelines);
        setSelectedPipelines(selectablePipelines);
      }
    };
    void fetchData();
    return () => {
      isSubscribed = false;
    };
  }, [userInfo]);

  // handle pipeline button click
  const handlePipelineClick = (group: string, pipelineName: string) => {
    const selected = structuredClone(selectedPipelines);
    selected[group][pipelineName] = !selectedPipelines[group][pipelineName];
    setSelectedPipelines(selected);
    setReactionsList(buildReactionsList(selected));
    if (setError) setError([]);
  };

  return (
    <Card className="panel">
      <Card.Body className="py-0">
        <center>
          {pipelinesListErrors &&
            pipelinesListErrors.map((error, idx) => <AlertBanner key={`pipeline-list-error-${idx}`}>{error}</AlertBanner>)}
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
                                tip={(<Markdown>{pipeline.description}</Markdown>) as unknown as string}
                              >
                                <Button
                                  variant=""
                                  className={`m-1 primary-btn ${selectedPipelines[`${pipeline.group}`]?.[`${pipeline.name}`] ? 'selected' : 'unselected'}`}
                                  onClick={() => handlePipelineClick(pipeline.group, pipeline.name)}
                                >
                                  <b>{pipeline.name}</b>
                                </Button>
                              </OverlayTipTop>
                            ) : (
                              <Button
                                variant=""
                                key={`${pipeline.group}_${pipeline.name}`}
                                className={`m-1 primary-btn ${selectedPipelines[`${pipeline.group}`]?.[`${pipeline.name}`] ? 'selected' : 'unselected'}`}
                                onClick={() => handlePipelineClick(pipeline.group, pipeline.name)}
                              >
                                <b>{pipeline.name}</b>
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
