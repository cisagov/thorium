import React from 'react';
import { Link } from 'react-router-dom';
import { Row } from 'react-bootstrap';

// project imports
import { EntityBrowseConfig, makeEntityFetcher } from './entityFetcher';
import {
  BrowsingCard,
  BrowsingContents,
  EntityGroups,
  EntityName,
  EntityNameWithIcon,
  EntitySecondary,
  EntitySubmitters,
  LinkFields,
} from '@entities/browsing/shared';
import CondensedEntityTags from '@components/tags/condensed/CondensedEntityTags';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';
import { Entities, entityLabel } from '@models/entities/entities';
import { PeSection } from '@models/entities/pe';

// spec: ../EntityBrowsing.spec.md

// Compose a compact size/entropy summary for the card, skipping fields the section didn't report.
const sectionSummary = (section: PeSection): string => {
  const meta = section.metadata.PeSection;
  const parts: string[] = [];
  if (meta.raw_size !== undefined) parts.push(`raw ${meta.raw_size}`);
  if (meta.virtual_size !== undefined) parts.push(`vsz ${meta.virtual_size}`);
  if (meta.entropy !== undefined) parts.push(`entropy ${meta.entropy}`);
  return parts.join(' · ');
};

const PeSectionListHeaders = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <EntityName>Name</EntityName>
          <EntitySecondary>Raw · Virtual · Entropy</EntitySecondary>
          <EntityGroups>Group(s)</EntityGroups>
          <EntitySubmitters>Submitter(s)</EntitySubmitters>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

interface PeSectionItemProps {
  section: PeSection;
}

const PeSectionItem: React.FC<PeSectionItemProps> = ({ section }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link
          to={`${getDetailsBasePathByEntity(Entities.PeSection)}/${section.id}`}
          state={{ peSection: section }}
          className="no-decoration"
        >
          <LinkFields>
            <EntityName>
              <EntityNameWithIcon entityId={section.id} hasImage={section.image != null}>
                {section.name}
              </EntityNameWithIcon>
            </EntityName>
            <EntitySecondary>{sectionSummary(section)}</EntitySecondary>
            <EntityGroups>
              <small>
                <i>{section.groups && section.groups.join(', ')}</i>
              </small>
            </EntityGroups>
            <EntitySubmitters>
              {section.submitter ? (
                <small>
                  <i>{section.submitter}</i>
                </small>
              ) : null}
            </EntitySubmitters>
          </LinkFields>
        </Link>
        {section.tags && Object.keys(section.tags).length > 1 && (
          <>
            <hr />
            <CondensedEntityTags resource={Entities.PeSection} tags={section.tags} />
          </>
        )}
      </BrowsingContents>
    </BrowsingCard>
  );
};

const PeSectionBrowsingConfig: EntityBrowseConfig<Entities.PeSection> = {
  docTitle: `${entityLabel(Entities.PeSection)}s · Thorium`,
  title: `${entityLabel(Entities.PeSection)}s`,
  typeLabel: entityLabel(Entities.PeSection),
  kind: Entities.PeSection,
  creatable: true,
  entityHeaders: <PeSectionListHeaders />,
  renderEntity: (entity) => <PeSectionItem section={entity} />,
  fetchEntities: makeEntityFetcher(Entities.PeSection),
};

export default PeSectionBrowsingConfig;
