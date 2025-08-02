import React, { Fragment } from 'react';
import { Col, Row } from 'react-bootstrap';

// project imports
import { filterIncludedTags, filterExcludedTags, FormattedFileInfoTagKeys, DangerTagKeys } from './utilities';
import { TagBadge } from './tags';
import { Entities, Tags } from '@models';

interface CondensedFileTagProps {
  tags: Tags; // tags to display in condensed non-editable view
}

const CondensedFileTags: React.FC<CondensedFileTagProps> = ({ tags }) => {
  const excludeTags = [...FormattedFileInfoTagKeys, 'RESULTS', 'ATT&CK', 'MBC', 'PARENT', 'SUBMITTER', ...DangerTagKeys];
  const dangerTags = filterIncludedTags(tags, DangerTagKeys);
  const generalTags = filterExcludedTags(tags, excludeTags);
  const fileInfoTags = filterIncludedTags(tags, FormattedFileInfoTagKeys);
  const attackTags = filterIncludedTags(tags, ['ATT&CK']);
  const mbcTags = filterIncludedTags(tags, ['MBC']);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const allTags = [dangerTags, attackTags, mbcTags, fileInfoTags, generalTags];

  return (
    <Fragment>
      <hr />
      <Row>
        <Col className="d-flex justify-content-center wrap">
          {Object.keys(dangerTags)
            .sort()
            .map((tagKey) =>
              Object.keys(dangerTags[tagKey])
                .sort()
                .map((tagValue) => (
                  <TagBadge
                    resource={Entities.File}
                    key={'FileInfo_' + tagValue}
                    tag={tagKey}
                    value={tagValue}
                    condensed={true}
                    action={'link'}
                  />
                )),
            )}
          {Object.keys(attackTags)
            .sort()
            .map((tagKey) =>
              Object.keys(attackTags[tagKey])
                .sort()
                .map((tagValue) => (
                  <TagBadge
                    resource={Entities.File}
                    key={'Attack_' + tagValue}
                    tag={'ATT&CK'}
                    value={tagValue}
                    condensed={true}
                    action={'link'}
                  />
                )),
            )}
          {Object.keys(mbcTags)
            .sort()
            .map((tagKey) =>
              Object.keys(mbcTags[tagKey])
                .sort()
                .map((tagValue) => (
                  <TagBadge
                    resource={Entities.File}
                    key={'MBC_' + tagValue}
                    tag={'MBC'}
                    value={tagValue}
                    condensed={true}
                    action={'link'}
                  />
                )),
            )}
          {Object.keys(fileInfoTags)
            .sort()
            .map((tagKey) =>
              Object.keys(fileInfoTags[tagKey])
                .sort()
                .map((tagValue) => (
                  <TagBadge
                    resource={Entities.File}
                    key={'FileInfo_' + tagValue}
                    tag={tagKey}
                    value={tagValue}
                    condensed={true}
                    action={'link'}
                  />
                )),
            )}
        </Col>
      </Row>
    </Fragment>
  );
};

export default CondensedFileTags;
