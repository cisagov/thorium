import { FaQuestionCircle } from 'react-icons/fa';

// project imports
import InfoHeader from '../shared/InfoHeader';
import { OverlayTipBottom } from '@components/shared/overlay/tips';

type EntityDetailsLabelProps = {
  label: string;
  tip: string;
};

const EntityDetailsLabel: React.FC<EntityDetailsLabelProps> = ({ label, tip }) => {
  return (
    <InfoHeader>
      {label}
      <OverlayTipBottom tip={tip}>
        <FaQuestionCircle />
      </OverlayTipBottom>
    </InfoHeader>
  );
};

export default EntityDetailsLabel;
