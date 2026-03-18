import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Row } from 'react-bootstrap';
import { Country } from 'country-list';

// project imports
import {
  BrowsingCard,
  BrowsingContents,
  EntityGroups,
  EntityName,
  EntityOrigin,
  EntitySubmitters,
  LinkFields,
} from '@entities/browsing/shared';
import BrowsingFilters from '@entities/browsing/filters/BrowsingFilters';
import EntityList from '@entities/browsing/EntityList';
import Page from '@components/pages/Page';
import CondensedEntityTags from '@components/tags/condensed/CondensedEntityTags';
import { useAuth } from '@utilities/auth';
import { Vendor } from '@models/entities/vendors';
import { Entities } from '@models/entities/entities';
import { Filters } from '@models/search';
import { listEntities } from '@thorpi/entities';

// get files using filters and and an optional cursor
const getVendors = async (filters: Filters, cursor: string | null, errorHandler: (error: string) => void) => {
  // reset cursor when filters have changed, caller must know this
  // get files list from API
  const listFilters = structuredClone(filters);
  listFilters.kinds = [Entities.Vendor];
  const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
  return {
    entitiesList: entityList,
    entitiesCursor: entityCursor,
  };
};

const VendorListHeaders = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <EntityName>Name</EntityName>
          <EntityOrigin>Country</EntityOrigin>
          <EntityGroups>Group(s)</EntityGroups>
          <EntitySubmitters>Submitter(s)</EntitySubmitters>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

interface VendorItemProps {
  vendor: Vendor; // device details
}

const VendorItem: React.FC<VendorItemProps> = ({ vendor }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`/vendor/${vendor.id}`} state={{ vendor: vendor }} className="no-decoration">
          <LinkFields>
            <EntityName>{vendor.name}</EntityName>
            <EntityOrigin>
              {vendor.metadata.Vendor.countries != undefined
                ? vendor.metadata.Vendor.countries
                    .map((country: Country) => {
                      return country.code;
                    })
                    .join(', ')
                    .substring(0, 30)
                : ''}
            </EntityOrigin>
            <EntityGroups>
              <i>
                {vendor.groups &&
                  (vendor.groups.toString().length > 75
                    ? vendor.groups.toString().replaceAll(',', ', ').substring(0, 75) + '...'
                    : vendor.groups.toString().replaceAll(',', ', '))}
              </i>
            </EntityGroups>
            <EntitySubmitters>
              {vendor.submitter ? (
                <i>{vendor.submitter.length > 75 ? vendor.submitter.substring(0, 75) + '...' : vendor.submitter}</i>
              ) : null}
            </EntitySubmitters>
          </LinkFields>
        </Link>
        {vendor.tags != undefined && <hr />}
        <Row>
          {vendor.tags && Object.keys(vendor.tags).length > 1 ? (
            <CondensedEntityTags resource={Entities.Vendor} tags={vendor.tags} />
          ) : null}
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

const VendorBrowsing = () => {
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState<Filters>({});
  const { userInfo } = useAuth();

  return (
    <Page title={`Vendors · Thorium`}>
      <BrowsingFilters
        title="Vendors"
        kind={Entities.Vendor}
        onChange={setFilters}
        groups={userInfo?.groups ? userInfo.groups : []}
        disabled={loading}
        creatable={true}
      />
      <EntityList
        type="Vendors"
        entityHeaders={<VendorListHeaders />}
        displayEntity={(device) => <VendorItem vendor={device} />}
        filters={filters}
        fetchEntities={getVendors}
        setLoading={setLoading}
        loading={loading}
      />
    </Page>
  );
};

export default VendorBrowsing;
