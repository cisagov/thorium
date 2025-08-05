import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Card, Col, Row } from 'react-bootstrap';

// project imports
import { BrowsingFilters, CondensedFileTags, EntityList, Page } from '@components';
import { getUniqueSubmissionGroups, useAuth } from '@utilities';
import { listFiles } from '@thorpi';
import { Filters } from '@models';

// get files using filters and an optional cursor
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

const FileListHeaders = () => {
  return (
    <Card className="basic-card panel">
      <Card.Body>
        <Row>
          <Col className="d-flex justify-content-center sha256-col">SHA256</Col>
          <Col className="d-flex justify-content-center submissions-col">Submissions</Col>
          <Col className="d-flex justify-content-center groups-col hide-element">Group(s)</Col>
          <Col className="d-flex justify-content-center submitters-col">Submitter(s)</Col>
        </Row>
      </Card.Body>
    </Card>
  );
};

interface FileItemProps {
  file: any; // file details
}

const FileItem: React.FC<FileItemProps> = ({ file }) => {
  return (
    <Card className="basic-card panel">
      <Card.Body>
        <Link to={`/file/${file.sha256}`} className="no-decoration">
          <Row className="highlight-card">
            <Col className="d-flex justify-content-center sha256-col sha256-hide">{file.sha256}</Col>
            <Col className="d-flex justify-content-center small-sha">{file.sha256.substr(0, 30) + '...'}</Col>
            <Col className="d-flex justify-content-center submissions-col">{file.submissions.length}</Col>
            <Col className="d-flex justify-content-center groups-col hide-element">
              <small>
                <i>
                  {getUniqueSubmissionGroups(file.submissions).toString().length > 75
                    ? getUniqueSubmissionGroups(file.submissions).toString().replaceAll(',', ', ').substring(0, 75) + '...'
                    : getUniqueSubmissionGroups(file.submissions).toString().replaceAll(',', ', ')}
                </i>
              </small>
            </Col>
            <Col className="d-flex justify-content-center submitters-col">
              {file.tags.submitter ? (
                <small>
                  <i>
                    {Object.keys(file.tags.submitter).toString().length > 75
                      ? Object.keys(file.tags.submitter).toString().replaceAll(',', ', ').substring(0, 75) + '...'
                      : Object.keys(file.tags.submitter).toString().replaceAll(',', ', ')}
                  </i>
                </small>
              ) : null}
            </Col>
          </Row>
        </Link>
        <Row>
          {Object.keys(file.tags).length > 1 || (Object.keys(file.tags).length == 1 && !file.tags.submitter) ? (
            <CondensedFileTags tags={file.tags} />
          ) : null}
        </Row>
      </Card.Body>
    </Card>
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
        displayEntity={(file) => <FileItem file={file} />}
        filters={filters}
        fetchEntities={getFiles}
        setLoading={setLoading}
        loading={loading}
      />
    </Page>
  );
};

export default FilesBrowsingContainer;
