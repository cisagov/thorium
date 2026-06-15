import type { FC } from 'react';
import { FaQuestionCircle } from 'react-icons/fa';

// project imports
import { InfoRow, HeaderCol, DetailCol, TriggerRow, TriggerIndent, TriggerField, TriggerValue } from './PipelineInfo.styled';
import FieldBadge from '@components/shared/badges/FieldBadge';
import SimpleSubtitle from '@components/shared/titles/SimpleSubtitle';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import type { EventTrigger, TagTrigger } from '@models/pipelines';

// spec: ./PipelineInfo.spec.md

interface TriggerDisplayProps {
  triggers: Record<string, EventTrigger>;
}

function isTagTrigger(trigger: EventTrigger): trigger is { Tag: TagTrigger } {
  return typeof trigger === 'object' && 'Tag' in trigger;
}

const TriggerDisplay: FC<TriggerDisplayProps> = ({ triggers }) => {
  const triggerNames = Object.keys(triggers);

  return (
    <>
      <InfoRow style={{ marginTop: '0.25rem' }}>
        <HeaderCol>
          <OverlayTipRight
            tip={`Automatic triggers that will cause this pipeline to run.
              Events can be configured to trigger when samples are initially uploaded or
              upon the creation of metadata tags.`}
          >
            <SimpleSubtitle>
              <b>Event Triggers</b> <FaQuestionCircle />
            </SimpleSubtitle>
          </OverlayTipRight>
        </HeaderCol>
        {triggerNames.length === 0 && (
          <DetailCol>
            <FieldBadge field={'None'} color={'#7e7c7c'} />
          </DetailCol>
        )}
      </InfoRow>
      {triggerNames.map((triggerName, idx) => {
        const trigger = triggers[triggerName];
        return (
          <div key={triggerName}>
            <TriggerRow>
              <TriggerIndent />
              <TriggerField>
                <em>Trigger Name:</em>
              </TriggerField>
              <TriggerValue>
                <FieldBadge field={triggerName} color={'#7e7c7c'} />
              </TriggerValue>
            </TriggerRow>
            {isTagTrigger(trigger) && (
              <>
                <TriggerRow>
                  <TriggerIndent />
                  <TriggerField>
                    <em>Trigger Type:</em>
                  </TriggerField>
                  <TriggerValue>
                    <FieldBadge field={'Tag'} color={'#7e7c7c'} />
                  </TriggerValue>
                </TriggerRow>
                <TriggerRow>
                  <TriggerIndent />
                  <TriggerField>
                    <em>Tag Types:</em>
                  </TriggerField>
                  <TriggerValue>
                    <FieldBadge field={trigger.Tag.tag_types} color={'#7e7c7c'} />
                  </TriggerValue>
                </TriggerRow>
                <TriggerRow>
                  <TriggerIndent />
                  <TriggerField>
                    <em>Required:</em>
                  </TriggerField>
                  <TriggerValue>
                    {Object.keys(trigger.Tag.required).length === 0 && <FieldBadge field={'None'} color={'#7e7c7c'} />}
                    {Object.keys(trigger.Tag.required)
                      .sort()
                      .map((key: string) =>
                        trigger.Tag.required[key].map((value: string) => (
                          <FieldBadge key={`${key}-${value}`} field={`${key}: ${value}`} color={'#7e7c7c'} />
                        )),
                      )}
                  </TriggerValue>
                </TriggerRow>
                <TriggerRow>
                  <TriggerIndent />
                  <TriggerField>
                    <em>Not:</em>
                  </TriggerField>
                  <TriggerValue>
                    {Object.keys(trigger.Tag.not).length === 0 && <FieldBadge field={'None'} color={'#7e7c7c'} />}
                    {Object.keys(trigger.Tag.not)
                      .sort()
                      .map((key: string) =>
                        trigger.Tag.not[key].map((value: string) => (
                          <FieldBadge key={`${key}-${value}`} field={`${key}: ${value}`} color={'#7e7c7c'} />
                        )),
                      )}
                  </TriggerValue>
                </TriggerRow>
              </>
            )}
            {trigger === 'NewSample' && (
              <TriggerRow>
                <TriggerIndent />
                <TriggerField>
                  <em>Trigger Type:</em>
                </TriggerField>
                <TriggerValue>
                  <FieldBadge field={'NewSample'} color={'#7e7c7c'} />
                </TriggerValue>
              </TriggerRow>
            )}
            {triggerNames.length - 1 !== idx && <hr className="tagshr" />}
          </div>
        );
      })}
    </>
  );
};

export default TriggerDisplay;
