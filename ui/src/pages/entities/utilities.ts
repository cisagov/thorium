import { listEntities } from '@thorpi';
import { Entities, Filters, Vendor } from '@models';

export const getAvailableVendors = async (updateVendors: (vendorsMap: { [key: string]: string }) => void) => {
  const filters: Filters = { kinds: [Entities.Vendor], limit: 10000 };
  const vendorsMap: { [key: string]: string } = {};
  const { entityList } = await listEntities(filters, console.log, true, null);
  if (entityList) {
    entityList.forEach((vendor: Vendor) => {
      vendorsMap[vendor.id] = vendor.name;
    });
  }
  updateVendors(vendorsMap);
};
