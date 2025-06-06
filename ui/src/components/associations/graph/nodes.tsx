import React from 'react';
import { Link } from 'react-router-dom';

// project imports
import {
  DangerTagKeys,
  FieldBadge,
  filterExcludedTags,
  filterIncludedTags,
  FormattedFileInfoTagKeys,
  Subtitle,
  TagBadge,
} from '@components';
import { getNodeName } from '../utilities';
import { Country } from 'country-list';
import { TreeNode, Vendor } from '@models';
import { Edge } from './edges';

// a cytoscape graph node
export interface Node {
  data: {
    id: string;
    label: string;
    diameter: number;
  };
  classes?: string[];
}

export function isNode(element: Edge | Node): element is Node {
  return 'id' in element?.data;
}

type NodeTagsProps = {
  node: TreeNode; // arbitrary Thorium node data
};

type FilteredTagsProp = {
  tags: any; // arbitrary tags data
};

const FilteredNodeTags: React.FC<FilteredTagsProp> = ({ tags }) => {
  const excludeTags = [...FormattedFileInfoTagKeys, 'RESULTS', 'ATT&CK', 'MBC', 'PARENT', 'SUBMITTER', ...DangerTagKeys];
  const dangerTags = filterIncludedTags(tags, DangerTagKeys);
  const generalTags = filterExcludedTags(tags, excludeTags);
  const fileInfoTags = filterIncludedTags(tags, FormattedFileInfoTagKeys);
  const attackTags = filterIncludedTags(tags, ['ATT&CK']);
  const mbcTags = filterIncludedTags(tags, ['MBC']);
  const allTags = [dangerTags, attackTags, mbcTags, fileInfoTags, generalTags];

  return (
    <>
      {allTags.map((tagGrouping) =>
        Object.keys(tagGrouping)
          .sort()
          .map((tagKey) =>
            Object.keys(tagGrouping[tagKey])
              .sort()
              .map((tagValue) => <TagBadge key={`${tagKey}_${tagValue}`} tag={tagKey} value={tagValue} condensed={true} action={'none'} />),
          ),
      )}
    </>
  );
};

export const NodeInfo: React.FC<NodeTagsProps> = ({ node }) => {
  // ignore empty nodes
  if (Object.keys(node).length === 0) {
    return <></>;
  }
  if (node.Sample) {
    const tags = node.Sample?.tags;
    return (
      <div className="m-2">
        <Subtitle>File Names</Subtitle>
        {getNodeName(node, 1000)}
        <hr />
        <Subtitle>SHA256</Subtitle>
        <Link to={`/file/${node.Sample.sha256}/`} target={'_blank'}>
          {node.Sample.sha256}
        </Link>
        <hr />
        <Subtitle className="mt-2">tags</Subtitle>
        {Object.keys(tags).length == 0 ? (
          <p className="ms-4 text"> N/A</p>
        ) : (
          <div className="d-flex justify-content-start wrap">
            <FilteredNodeTags tags={tags} />
          </div>
        )}
        <hr />
      </div>
    );
  } else if (node.Repo) {
    const tags = node.Repo.tags;
    return (
      <div className="m-2">
        <Subtitle>REPO</Subtitle>
        <Link to={`/repo/${node.Repo.url}`} target={'_blank'}>
          {node.Repo.url}
        </Link>
        <hr />
        <Subtitle className="mt-2">tags</Subtitle>
        {Object.keys(tags).length == 0 ? (
          <p className="ms-4 text"> N/A</p>
        ) : (
          <div className="d-flex justify-content-start wrap">
            <FilteredNodeTags tags={tags} />
          </div>
        )}
      </div>
    );
  } else if (node.Tag) {
    const tags = node.Tag.tags;
    return (
      <div className="m-2">
        <Subtitle className="mb-4">Tag(s)</Subtitle>
        <div style={{ minWidth: '200px' }}>
          {Object.keys(tags).length == 0 ? (
            <p className="ms-3 text"> N/A</p>
          ) : (
            <div className="d-flex justify-content-start wrap">
              {Object.keys(tags).map((tagKey) => (
                <TagBadge key={`${tagKey}_${tags[tagKey]}`} tag={tagKey} value={`${tags[tagKey]}`} condensed={true} action={'none'} />
              ))}
            </div>
          )}
        </div>
      </div>
    );
  } else if (node.Entity) {
    const tags = node.Entity.tags;
    // add any Entity specific metadata we want to display
    const entity = node.Entity;
    const buildEntitySummary = () => {
      if (entity?.kind == 'Device') {
        return (
          <>
            <Subtitle className="mt-2">Vendor(s)</Subtitle>
            {entity.metadata.Device.vendors.length > 0 ? (
              <FieldBadge
                color="Gray"
                noNull={true}
                field={entity.metadata.Device.vendors.map((vendor: Vendor) => `${vendor.name} (${vendor.id})`)}
              />
            ) : (
              <small>N/A</small>
            )}
            <hr />
            <Subtitle className="mt-2">
              {`Critical System: `}
              {entity.metadata.Device.critical_system ? (
                <FieldBadge color="DarkRed" noNull={true} field={'yes'} />
              ) : (
                <FieldBadge color="Gray" noNull={true} field={'no'} />
              )}
            </Subtitle>
            {entity.metadata.Device.critical_sectors.length > 0 && (
              <Subtitle className="mt-2">
                {`Critical Sectors: `}
                <FieldBadge color="DarkRed" noNull={true} field={entity.metadata.Device.critical_sectors} />
              </Subtitle>
            )}
            <Subtitle className="mt-2">
              {`Sensitive Location: `}
              {entity.metadata.Device.sensitive_location ? (
                <FieldBadge color="DarkRed" noNull={true} field={'yes'} />
              ) : (
                <FieldBadge color="Gray" noNull={true} field={'no'} />
              )}
            </Subtitle>
          </>
        );
      } else if (entity?.kind == 'Vendor') {
        return (
          <>
            <Subtitle className="mt-2">Countries</Subtitle>
            {entity.metadata.Vendor.countries.length > 0 ? (
              <FieldBadge color="Gray" noNull={true} field={entity.metadata.Vendor.countries.map((country: Country) => country.name)} />
            ) : (
              <small>N/A</small>
            )}
            <hr />
            {entity.metadata.Vendor.critical_sectors.length > 0 && (
              <Subtitle className="mt-2">
                {`Critical Sectors: `}
                <FieldBadge color="DarkRed" noNull={true} field={entity.metadata.Vendor.critical_sectors} />
              </Subtitle>
            )}
          </>
        );
      }
      return <></>;
    };
    return (
      <div className="m-2">
        <Subtitle>{node.Entity.kind}</Subtitle>
        <Link to={`/${node.Entity.kind.toLowerCase()}/${node.Entity.id}`} target={'_blank'}>
          {`${node.Entity.name} (${node.Entity.id})`}
        </Link>
        <hr />
        {buildEntitySummary()}
        <hr />
        <Subtitle className="mt-2">Tags</Subtitle>
        {Object.keys(tags).length == 0 ? (
          <p className="ms-4 text"> N/A</p>
        ) : (
          <div className="d-flex justify-content-start wrap">
            <FilteredNodeTags tags={tags} />
          </div>
        )}
      </div>
    );
  } else {
    <div className="m-2">{JSON.stringify(node, null, 2)}</div>;
  }
};
