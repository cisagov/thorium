import React, { useState } from 'react';
import { Button, Modal } from 'react-bootstrap';

// project imports
import { OverlayTipBottom } from '@components';
import { FormattedFileInfoTagKeys, DangerTagKeys } from './utilities';
import { Entities } from '@models';

interface TagBadgeProps {
  tag: string; // tag key string
  value: string; // key value
  condensed: boolean; // show condensed view of TLP tag that hides the key "TLP"
  action: string; // onclick action keyword
  resource?: Entities;
}

const TagBadge: React.FC<TagBadgeProps> = ({ tag, value, condensed, action, resource }) => {
  const [showRedirectModal, setShowRedirectModal] = useState(false);
  let badgeClass = '';
  let tagText = '';

  const upperTag = tag.toUpperCase();
  // format traffic light protocol tags
  if (upperTag == 'TLP') {
    // on details page, only print value because TLP is in a different col
    if (!condensed) {
      tagText = value.toUpperCase();
    } else {
      tagText = `TLP: ${value.toUpperCase()}`;
    }
    switch (value.toUpperCase()) {
      case 'RED':
        badgeClass = 'tlp-red-btn';
        break;
      case 'AMBER':
        badgeClass = 'tlp-amber-btn';
        break;
      case 'AMBER+STRICT':
        badgeClass = 'tlp-amber-btn';
        break;
      case 'GREEN':
        badgeClass = 'tlp-green-btn';
        break;
      case 'WHITE':
        badgeClass = 'tlp-clear-btn';
        break;
      case 'CLEAR':
        badgeClass = 'tlp-clear-btn';
        break;
    }
  } else if (upperTag == 'RESULTS') {
    badgeClass = 'general-tag';
    tagText = `${tag}: ${value}`;
  } else if (upperTag == 'ATT&CK') {
    badgeClass = 'attack-tag';
    tagText = `${value}`;
  } else if (upperTag == 'MBC') {
    badgeClass = 'mbc-tag';
    tagText = `${value}`;
  } else if (FormattedFileInfoTagKeys.includes(upperTag)) {
    badgeClass = 'info-tag';
    tagText = `${tag}: ${value}`;
  } else {
    if (DangerTagKeys.includes(tag.toUpperCase())) {
      badgeClass = 'danger-tag';
    } else {
      badgeClass = 'other-tag';
    }
    tagText = `${tag}: ${value}`;
  }
  // returned rendered component
  if (action == 'scroll') {
    const scrollToResult = (value: string) => {
      const element = document.getElementById(`results-tab-${value}`);
      if (element) {
        element.scrollIntoView();
      }
    };
    return (
      <OverlayTipBottom tip={`Click to jump to ${value} results`}>
        <div className={`${badgeClass} ms-1 mb-1 tag-item clickable tags-hide`} onClick={() => scrollToResult(value)}>
          {tagText}
        </div>
        <div className={`${badgeClass} ms-1 mb-1 tag-item clickable short-tag`} onClick={() => scrollToResult(value)}>
          {tagText.length > 30 ? tagText.substring(0, 30) + '...' : tagText}
        </div>
      </OverlayTipBottom>
    );
    // link to external mitre docs for Att&ck tags
  } else if (action == 'docs' && upperTag == 'ATT&CK') {
    const tactic = value.split(' ');
    const attackID = tactic.at(-1)?.split('.')[0];
    const attackSubID = tactic.at(-1)?.split('.').at(1);
    let redirectURL = '';
    if (attackSubID != undefined) {
      redirectURL = `https://attack.mitre.org/techniques/${attackID}/${attackSubID}/`;
    } else {
      redirectURL = `https://attack.mitre.org/techniques/${attackID}/`;
    }
    // on click function to redirect to external URL
    const redirectToExternal = () => {
      window.open(redirectURL, '_blank');
    };
    return (
      <>
        <Modal show={showRedirectModal} onHide={() => setShowRedirectModal(false)}>
          <Modal.Header closeButton>
            <h3>Navigate to an external site?</h3>
          </Modal.Header>
          <Modal.Body className="d-flex justify-content-center">
            <i>{redirectURL}</i>
          </Modal.Body>
          <Modal.Footer className="d-flex justify-content-center">
            {/* @ts-ignore */}
            <Button
              variant=""
              className="warning-btn"
              onClick={() => {
                redirectToExternal();
                setShowRedirectModal(false);
              }}
            >
              Confirm
            </Button>
          </Modal.Footer>
        </Modal>
        <OverlayTipBottom tip={`Click to see mitre documentation on this technique: ${tagText}`}>
          <a className="no-decoration" onClick={() => setShowRedirectModal(true)}>
            <div className={`${badgeClass} ms-1 mb-1 tag-item clickable tags-hide`}>{tagText}</div>
            <div className={`${badgeClass} ms-1 mb-1 tag-item clickable short-tag`}>
              {tagText.length > 30 ? tagText.substring(0, 30) + '...' : tagText}
            </div>
          </a>
        </OverlayTipBottom>
      </>
    );
    // link to external mitre docs for MBC tags
  } else if (action == 'docs' && upperTag == 'MBC') {
    const splitIndex = value.lastIndexOf(' ');
    const identifier = value.slice(splitIndex);
    const splitText = value.slice(0, splitIndex).split('::');
    const behavior = splitText[0].replaceAll(' ', '-').toLowerCase();
    const method = splitText[1].replaceAll(' ', '-').toLowerCase();
    let redirectURL = '';
    if (!identifier.includes('C')) {
      redirectURL = `https://github.com/MBCProject/mbc-markdown/tree/v3.0/${behavior}/${method}.md`;
    } else {
      redirectURL = `https://github.com/MBCProject/mbc-markdown/tree/v3.0/micro-behaviors/${behavior}/${method}.md`;
    }
    // on click function to redirect to external URL
    const redirectToExternal = () => {
      window.open(redirectURL, '_blank');
    };

    return (
      <>
        <Modal show={showRedirectModal} onHide={() => setShowRedirectModal(false)}>
          <Modal.Header closeButton>
            <h3>Navigate to an external site?</h3>
          </Modal.Header>
          <Modal.Body className="d-flex justify-content-center">
            <i>{redirectURL}</i>
          </Modal.Body>
          <Modal.Footer className="d-flex justify-content-center">
            <Button
              variant=""
              className="warning-btn"
              onClick={() => {
                redirectToExternal();
                setShowRedirectModal(false);
              }}
            >
              Confirm
            </Button>
          </Modal.Footer>
        </Modal>
        <OverlayTipBottom tip={`Click to see mitre documentation on this behavior: ${tagText}`}>
          <a className="no-decoration" onClick={() => setShowRedirectModal(true)}>
            <div className={`${badgeClass} ms-1 mb-1 tag-item clickable tags-hide`}>{tagText}</div>
            <div className={`${badgeClass} ms-1 mb-1 tag-item clickable short-tag`}>
              {tagText.length > 30 ? tagText.substring(0, 30) + '...' : tagText}
            </div>
          </a>
        </OverlayTipBottom>
      </>
    );
  } else if (action == 'link') {
    // built the URL search query params from the tag key and value
    return (
      <OverlayTipBottom tip={`Click to browse ${resource}s with tag: ${tagText}`}>
        <a
          className="no-decoration"
          onClick={() => {
            if (resource === undefined) {
              console.log('Error: No resource type provided for link');
              return;
            }
            // we are already browsing and want to append tags to current search params
            if (window.location.pathname.startsWith(resource.toLowerCase(), 1)) {
              const query = new URLSearchParams(window.location.search);
              query.append(`tags[${tag}]`, value);
              window.location.href = `/${resource.toLowerCase()}s?${query.toString()}`;
            } else {
              const query = new URLSearchParams();
              query.append('limit', '10');
              query.append(`tags[${tag}]`, value);
              window.location.href = `/${resource.toLowerCase()}s?${query.toString()}`;
            }
          }}
        >
          <div className={`${badgeClass} ms-1 mb-1 tag-item clickable tags-hide`}>{tagText}</div>
          <div className={`${badgeClass} ms-1 mb-1 tag-item clickable short-tag`}>
            {tagText.length > 30 ? tagText.substring(0, 30) + '...' : tagText}
          </div>
        </a>
      </OverlayTipBottom>
    );
  } else {
    return (
      <div>
        <div className={`${badgeClass} ms-1 mb-1 tag-item tags-hide`}>{tagText}</div>
        <div className={`${badgeClass} ms-1 mb-1 tag-item short-tag`}>
          {tagText.length > 30 ? tagText.substring(0, 30) + '...' : tagText}
        </div>
      </div>
    );
  }
};

export { TagBadge };
