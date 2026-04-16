import React from 'react';
import { Link } from 'react-router-dom';
import { Col, Row } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { EntityBrowseConfig } from './config';
import { BrowsingCard, BrowsingContents, LinkFields } from '@entities/browsing/shared';
import CondensedFileTags from '@components/tags/condensed/CondensedFileTags';
import { getUniqueFileNames } from '@utilities/files';
import { getUniqueSubmissionGroups } from '@utilities/groups';
import { listFiles } from '@thorpi/files';
import { Filters } from '@models/search';
import { Sample } from '@models/entities/files';
import { Entities } from '@models/entities/entities';
import { scaling } from '@styles';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';

// get files using filters and and an optional cursor
const getFiles = async (filters: Filters, existingCursor: string | null) => {
  // get files list from API
  const { files, cursor } = await listFiles(
    filters,
    console.log,
    true, // details bool
    existingCursor,
  );
  return {
    entitiesList: files,
    entitiesCursor: cursor,
  };
};

const Name = styled(Col)`
  white-space: pre-wrap;
  word-break: break-all;
  min-width: 600px;
  color: var(--thorium-text);
  @media (max-width: ${scaling.md}) {
    min-width: 70%;
  }
  @media (max-width: ${scaling.sm}) {
    min-width: 300px;
  }
`;

const Submissions = styled(Col)`
  min-width: 100px;
  text-align: center;
  color: var(--thorium-text);
  @media (max-width: ${scaling.xxl}) {
    display: none !important;
  }
`;

const Submitters = styled(Col)`
  flex-wrap: wrap;
  text-align: center;
  min-width: 150px;
  color: var(--thorium-text);
  @media (max-width: ${scaling.xl}) {
    display: none !important;
  }
`;

const Groups = styled(Col)`
  flex-wrap: wrap;
  text-align: center;
  min-width: 150px;
  color: var(--thorium-text);
  @media (max-width: ${scaling.lg}) {
    display: none !important;
  }
`;

const Sha256 = styled.div`
  font-size: 0.8rem;
  font-style: italic;
`;

const FileListHeaders = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <Name>File</Name>
          <Submissions>Submissions</Submissions>
          <Groups>Group(s)</Groups>
          <Submitters>Submitter(s)</Submitters>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

interface FileItemProps {
  file: Sample; // file details
  excludeKeys: string[]; // tag keys to exclude when displaying condensed tags
}

const FileItem: React.FC<FileItemProps> = ({ file, excludeKeys }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`${getDetailsBasePathByEntity(Entities.File)}/${file.sha256}`} className="no-decoration">
          <LinkFields>
            <Name>{getUniqueFileNames(file.submissions)}</Name>
            <Submissions>{file.submissions.length}</Submissions>
            <Groups>
              <small>
                <i>
                  {getUniqueSubmissionGroups(file.submissions).toString().length > 75
                    ? getUniqueSubmissionGroups(file.submissions).toString().replaceAll(',', ', ').substring(0, 75) + '...'
                    : getUniqueSubmissionGroups(file.submissions).toString().replaceAll(',', ', ')}
                </i>
              </small>
            </Groups>
            <Submitters>
              {file.tags.submitter ? (
                <small>
                  <i>
                    {Object.keys(file.tags.submitter).toString().length > 75
                      ? Object.keys(file.tags.submitter).toString().replaceAll(',', ', ').substring(0, 75) + '...'
                      : Object.keys(file.tags.submitter).toString().replaceAll(',', ', ')}
                  </i>
                </small>
              ) : null}
            </Submitters>
          </LinkFields>
        </Link>
        <Sha256 className="mt-3 mb-2">{file.sha256}</Sha256>
        <Row>
          {Object.keys(file.tags).length > 1 || (Object.keys(file.tags).length == 1 && !file.tags.submitter) ? (
            <CondensedFileTags tags={file.tags} excludeKeys={excludeKeys} />
          ) : null}
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

const FileBrowsingConfig: EntityBrowseConfig<Entities.File> = {
  docTitle: 'Files · Thorium',
  title: 'Files',
  typeLabel: '',
  kind: Entities.File,
  creatable: true,
  entityHeaders: <FileListHeaders />,
  renderEntity: (entity, filters) => <FileItem file={entity} excludeKeys={filters['hideTags'] ? filters['hideTags'] : []} />,
  fetchEntities: getFiles,
};

export default FileBrowsingConfig;
