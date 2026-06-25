import { useRef, useState } from 'react';
import type { FC } from 'react';
import { useNavigate } from 'react-router-dom';

// project imports
import ImageInfo from './ImageInfo';
import type { ImageInfoHandle } from './ImageInfo';
import { BanWarningIcon, DeleteConfirmModal, HeaderActions, HeaderBtn } from '@components/shared/browsing';
import { ButtonVariant } from '@components/shared/buttons';
import { OverlayTipBottom, OverlayTipLeft, OverlayTipRight } from '@components/shared/overlay/tips';
import { deleteImage } from '@thorpi/images';
import { useAuth } from '@utilities/auth';
import { generateCopyName } from '@utilities/naming';
import { canDeleteImage, canDevelopAnyInGroup, canModifyImage } from '@utilities/permissions';
import type { Group } from '@models/groups';
import type { Image } from '@models/images';
import { updateURLSection } from '@utilities/url';
import { toast } from 'react-toastify';
import { FaLink } from 'react-icons/fa';
import { AccordionItem, AccordionHeader, AccordionBody } from '@components/shared/accordion/Accordion';

interface ImageAccordionItemProps {
  image: Image;
  images: Image[];
  groups: Record<string, Group>;
  setImages: (images: Image[]) => void;
  // Full list reload (with spinner) used after this image is deleted.
  onRefresh: () => void;
  onExpand: (key: string) => void;
}

const ImageAccordionItem: FC<ImageAccordionItemProps> = ({ image, images, groups, setImages, onRefresh, onExpand }) => {
  const navigate = useNavigate();
  const { userInfo } = useAuth();
  const [inEditMode, setEditMode] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleteError, setDeleteError] = useState('');
  const imageInfoRef = useRef<ImageInfoHandle>(null);

  // gate edit/delete/create against the backend image authorization predicates
  const group = groups[image.group];
  const userCanModify = !!userInfo && !!group && canModifyImage(image, group, userInfo);
  const userCanDelete = !!userInfo && !!group && canDeleteImage(image, group, userInfo);
  const userCanCreateImage = !!userInfo && !!group && canDevelopAnyInGroup(group, userInfo);

  const hasBans = image.bans && Object.keys(image.bans).length > 0;

  const handleCloseDeleteModal = () => {
    setShowDeleteModal(false);
    setDeleteError('');
  };

  const handleDelete = async () => {
    if (await deleteImage(image.group, image.name, setDeleteError)) {
      // Reload the whole list (deleting changes the set of images); onRefresh drives the spinner.
      onRefresh();
      handleCloseDeleteModal();
    }
  };

  const id = `${image.name}--${image.group}`;

  const handleLink = (e: React.MouseEvent<HTMLAnchorElement, MouseEvent>) => {
    e.stopPropagation(); //don't close accordion
    updateURLSection(id, '');
    void navigator.clipboard.writeText(window.location.href);
    const notify = () => toast(`Copied "${window.location.href}" to clipboard!`);
    notify();
    onExpand(id);
  };

  return (
    <AccordionItem eventKey={id}>
      <AccordionHeader>
        <div className="accordion-item-name">
          <div className="text">
            <OverlayTipBottom tip={`Copy image link to clipboard`}>
              <a className="title-link" onClick={handleLink}>
                <FaLink className="title-link-no-color me-3" size={12} />
              </a>
            </OverlayTipBottom>
            {image.name}
          </div>
        </div>
        <div className="accordion-item-relation" />
        <div className="accordion-item-ownership">
          <OverlayTipLeft tip={`This image is owned by the ${image.group} group.`}>
            <small>
              <i>{image.group}</i>
            </small>
          </OverlayTipLeft>
        </div>
        <div className="accordion-item-status">
          {hasBans && (
            <OverlayTipRight tip="This image has active bans and cannot be used.">
              <BanWarningIcon />
            </OverlayTipRight>
          )}
        </div>
        <HeaderActions onClick={(e) => e.stopPropagation()}>
          {!inEditMode && userCanCreateImage && (
            <OverlayTipBottom tip={`Create a new image using ${image.name} as a template.`}>
              <HeaderBtn
                $variant={ButtonVariant.Ok}
                data-testid="header-btn-copy"
                onClick={() => {
                  const allNames = images.map((img) => img.name);
                  const copyName = generateCopyName(image.name, allNames);
                  void navigate('/create/image', { state: { ...image, name: copyName } });
                }}
              >
                Copy
              </HeaderBtn>
            </OverlayTipBottom>
          )}
          {!inEditMode && userCanModify && (
            <OverlayTipBottom tip="Edit this image.">
              <HeaderBtn
                $variant={ButtonVariant.Secondary}
                data-testid="header-btn-edit"
                onClick={() => {
                  setEditMode(true);
                  onExpand(id);
                }}
              >
                Edit
              </HeaderBtn>
            </OverlayTipBottom>
          )}
          {inEditMode && userCanModify && (
            <OverlayTipBottom tip="Submit pending updates.">
              <HeaderBtn $variant={ButtonVariant.Ok} onClick={() => imageInfoRef.current?.handleUpdate()}>
                Accept
              </HeaderBtn>
            </OverlayTipBottom>
          )}
          {inEditMode && userCanModify && (
            <OverlayTipBottom tip="Discard pending changes.">
              <HeaderBtn $variant={ButtonVariant.Secondary} onClick={() => setEditMode(false)}>
                Discard
              </HeaderBtn>
            </OverlayTipBottom>
          )}
          {userCanDelete && (
            <OverlayTipBottom tip="Delete this image.">
              <HeaderBtn $variant={ButtonVariant.Warning} data-testid="header-btn-delete" onClick={() => setShowDeleteModal(true)}>
                Delete
              </HeaderBtn>
            </OverlayTipBottom>
          )}
        </HeaderActions>
      </AccordionHeader>
      <AccordionBody>
        <ImageInfo
          ref={imageInfoRef}
          images={images}
          image={image}
          groups={groups}
          setImages={setImages}
          inEditMode={inEditMode}
          onExitEditMode={() => setEditMode(false)}
          userCanModify={userCanModify}
        />
      </AccordionBody>
      <DeleteConfirmModal
        show={showDeleteModal}
        onHide={handleCloseDeleteModal}
        onConfirm={() => void handleDelete()}
        itemName={image.name}
        itemType="image"
        error={deleteError}
      />
    </AccordionItem>
  );
};

export default ImageAccordionItem;
