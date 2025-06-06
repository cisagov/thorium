import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Col, Row } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { BrowsingFilters, CondensedFileTags, EntityList, Page, LinkFields, BrowsingCard, BrowsingContents } from '@components';
import { getUniqueFileNames, getUniqueSubmissionGroups, useAuth } from '@utilities';
import { listFiles } from '@thorpi';
import { Filters } from '@models';
import { scaling } from '@styles';

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
  file: any; // file details
  excludeKeys: string[]; // tag keys to exclude when displaying condensed tags
}

const FileItem: React.FC<FileItemProps> = ({ file, excludeKeys }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`/file/${file.sha256}`} className="no-decoration">
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

const FilesBrowsingContainer = () => {
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState<Filters>({});
  const { userInfo } = useAuth();

  return (
    <Page title={`Files · Thorium`}>
      <BrowsingFilters title="Files" onChange={setFilters} groups={userInfo ? userInfo.groups : []} disabled={loading} />
      <EntityList
        type="Files"
        entityHeaders={<FileListHeaders />}
        displayEntity={(file) => <FileItem file={file} excludeKeys={filters['hideTags'] ? filters['hideTags'] : []} />}
        filters={filters}
        fetchEntities={getFiles}
        setLoading={setLoading}
        loading={loading}
      />
    </Page>
  );
};

export default FilesBrowsingContainer;
