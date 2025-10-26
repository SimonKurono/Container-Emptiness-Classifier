"""Automated order placement system for multiple vendors."""

import asyncio
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class OrderStatus(Enum):
    """Order status enumeration."""
    
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


@dataclass
class OrderItem:
    """Order item data model."""
    
    product_id: str
    product_name: str
    quantity: int
    price: float
    total: float


@dataclass
class Product:
    """Product data model."""
    
    product_id: str
    name: str
    price: float
    vendor: str
    asin: Optional[str] = None
    upc: Optional[str] = None
    sku: Optional[str] = None


@dataclass
class Order:
    """Order data model."""
    
    order_id: str
    user_id: str
    items: List[OrderItem]
    subtotal: float
    tax: float
    shipping: float
    total: float
    status: OrderStatus
    vendor: str
    delivery_address: Optional[Dict] = None
    vendor_order_id: Optional[str] = None
    tracking_number: Optional[str] = None
    created_at: datetime = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class PlaceOrderRequest:
    """Order placement request."""
    
    product_id: str
    quantity: int
    user_id: str
    delivery_address: Optional[Dict] = None
    confirm: bool = False


@dataclass
class PlaceOrderResponse:
    """Order placement response."""
    
    success: bool
    order: Optional[Order] = None
    error: Optional[str] = None
    message: Optional[str] = None


PRODUCTS = {
    "prod_001": Product("prod_001", "Wireless Bluetooth Headphones", 99.99, "Amazon", asin="B08XYZ123"),
    "prod_002": Product("prod_002", "Organic Coffee Beans", 15.99, "Instacart", sku="COFFEE001"),
    "prod_003": Product("prod_003", "Smart Watch", 199.99, "Walmart", upc="123456789012"),
    "prod_004": Product("prod_004", "Grocery Essentials Pack", 45.99, "Instacart", sku="GROCERY001"),
    "prod_005": Product("prod_005", "Laptop Stand", 29.99, "Amazon", asin="B09ABC456"),
}


class OrderSystem:
    """Order management system with automatic vendor integration."""
    
    def __init__(self):
        """Initialize order system with configuration."""
        self.orders: Dict[str, Order] = {}
        self.state_tax_rates = {
            "CA": 0.0725, 
            "NY": 0.0800, 
            "TX": 0.0625, 
            "FL": 0.0600, 
            "WA": 0.0650
        }
        self.default_tax_rate = 0.07
        self.shipping_cost = 5.99
        self.free_shipping_threshold = 35.00
    
    def get_product(self, product_id: str) -> Optional[Product]:
        """Retrieve product by ID."""
        return PRODUCTS.get(product_id)
    
    def _calculate_tax(self, subtotal: float, delivery_address: Optional[Dict]) -> float:
        """Calculate tax based on delivery state."""
        state = delivery_address.get("state", "CA") if delivery_address else "CA"
        tax_rate = self.state_tax_rates.get(state, self.default_tax_rate)
        return round(subtotal * tax_rate, 2)
    
    def _calculate_shipping(self, subtotal: float) -> float:
        """Calculate shipping cost."""
        return 0.0 if subtotal >= self.free_shipping_threshold else self.shipping_cost
    
    async def _place_vendor_order(self, order: Order, product: Product) -> Dict:
        """Place order with vendor API."""
        vendor = product.vendor.lower()
        
        print(f"Placing {vendor} order for: {product.name}")
        
        if vendor == "amazon":
            vendor_order_id = f"AMZ-{uuid.uuid4().hex[:12].upper()}"
            estimated_delivery = "3-5 business days"
        elif vendor == "instacart":
            vendor_order_id = f"IC-{uuid.uuid4().hex[:12].upper()}"
            estimated_delivery = "same day (2-4 hours)"
        elif vendor == "walmart":
            vendor_order_id = f"WM-{uuid.uuid4().hex[:12].upper()}"
            estimated_delivery = "2-4 business days"
        else:
            return {"success": False, "error": f"Unsupported vendor: {vendor}"}
        
        return {
            "success": True,
            "vendor_order_id": vendor_order_id,
            "estimated_delivery": estimated_delivery
        }
    
    async def place_order(self, request: PlaceOrderRequest) -> PlaceOrderResponse:
        """Place an order for a product."""
        try:
            product = self.get_product(request.product_id)
            if not product:
                return PlaceOrderResponse(
                    success=False,
                    error=f"Product {request.product_id} not found",
                    message="Product not found in database"
                )
            
            if not request.confirm:
                return PlaceOrderResponse(
                    success=False,
                    error="Order not confirmed",
                    message="User must confirm order before placement"
                )
            
            item_price = product.price
            item_total = item_price * request.quantity
            
            order_item = OrderItem(
                product_id=product.product_id,
                product_name=product.name,
                quantity=request.quantity,
                price=item_price,
                total=item_total
            )
            
            subtotal = item_total
            tax = self._calculate_tax(subtotal, request.delivery_address)
            shipping = self._calculate_shipping(subtotal)
            total = subtotal + tax + shipping
            
            order_id = str(uuid.uuid4())
            order = Order(
                order_id=order_id,
                user_id=request.user_id,
                items=[order_item],
                subtotal=subtotal,
                tax=tax,
                shipping=shipping,
                total=total,
                status=OrderStatus.PENDING,
                vendor=product.vendor,
                delivery_address=request.delivery_address
            )
            
            vendor_result = await self._place_vendor_order(order, product)
            
            if vendor_result["success"]:
                order.vendor_order_id = vendor_result["vendor_order_id"]
                order.status = OrderStatus.CONFIRMED
                order.updated_at = datetime.utcnow()
                
                self.orders[order_id] = order
                
                return PlaceOrderResponse(
                    success=True,
                    order=order,
                    message=f"Order placed successfully with {product.vendor}!"
                )
            else:
                return PlaceOrderResponse(
                    success=False,
                    error=vendor_result.get("error", "Unknown vendor error"),
                    message="Failed to place order with vendor"
                )
                
        except Exception as e:
            return PlaceOrderResponse(
                success=False,
                error=str(e),
                message="Internal error while placing order"
            )
    
    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """Retrieve order by ID."""
        return self.orders.get(order_id)
    
    def get_user_orders(
        self, 
        user_id: str, 
        status: Optional[OrderStatus] = None, 
        limit: int = 50
    ) -> List[Order]:
        """Retrieve orders for a specific user."""
        user_orders = [order for order in self.orders.values() if order.user_id == user_id]
        
        if status:
            user_orders = [o for o in user_orders if o.status == status]
        
        user_orders.sort(key=lambda o: o.created_at, reverse=True)
        return user_orders[:limit]
    
    async def cancel_order(self, order_id: str, user_id: str, reason: str = "") -> Dict:
        """Cancel an order."""
        order = self.orders.get(order_id)
        
        if not order:
            return {"success": False, "error": "Order not found"}
        
        if order.user_id != user_id:
            return {"success": False, "error": "Unauthorized"}
        
        if order.status in [OrderStatus.SHIPPED, OrderStatus.DELIVERED]:
            return {"success": False, "error": "Cannot cancel shipped/delivered order"}
        
        print(f"Cancelling {order.vendor} order: {order.vendor_order_id}")
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        self.orders[order_id] = order
        
        return {
            "success": True,
            "message": f"Order {order_id} cancelled successfully",
            "refund_amount": order.total
        }
    
    def get_user_spending_summary(self, user_id: str) -> Dict:
        """Get spending summary for a user."""
        user_orders = self.get_user_orders(user_id)
        
        total_spent = sum(o.total for o in user_orders if o.status != OrderStatus.CANCELLED)
        total_orders = len([o for o in user_orders if o.status != OrderStatus.CANCELLED])
        
        return {
            "user_id": user_id,
            "total_orders": total_orders,
            "total_spent": round(total_spent, 2),
            "average_order_value": round(total_spent / total_orders, 2) if total_orders > 0 else 0.0,
            "orders_by_status": {
                status.value: len([o for o in user_orders if o.status == status])
                for status in OrderStatus
            }
        }


order_system = OrderSystem()


async def place_order(request: PlaceOrderRequest) -> PlaceOrderResponse:
    """Place an order for a product."""
    return await order_system.place_order(request)


def get_order_by_id(order_id: str) -> Optional[Order]:
    """Get order by ID."""
    return order_system.get_order_by_id(order_id)


def get_user_orders(
    user_id: str, 
    status: Optional[OrderStatus] = None, 
    limit: int = 50
) -> List[Order]:
    """Get orders for a specific user."""
    return order_system.get_user_orders(user_id, status, limit)


async def cancel_order(order_id: str, user_id: str, reason: str = "") -> Dict:
    """Cancel an order."""
    return await order_system.cancel_order(order_id, user_id, reason)


def get_user_spending_summary(user_id: str) -> Dict:
    """Get spending summary for a user."""
    return order_system.get_user_spending_summary(user_id)


async def demo_automatic_orders():
    """Demo automatic order placement."""
    print("AUTOMATIC ORDER PLACEMENT DEMO")
    print("=" * 50)
    
    print("\nTesting Amazon Order...")
    amazon_request = PlaceOrderRequest(
        product_id="prod_001",
        quantity=2,
        user_id="demo_user",
        delivery_address={"street": "123 Main St", "city": "San Francisco", "state": "CA", "zip": "94102"},
        confirm=True
    )
    
    response = await place_order(amazon_request)
    if response.success:
        print(f"Amazon order placed: {response.order.order_id}")
        print(f"   Total: ${response.order.total:.2f}")
    else:
        print(f"Amazon order failed: {response.error}")
    
    print("\nTesting Instacart Order...")
    instacart_request = PlaceOrderRequest(
        product_id="prod_002",
        quantity=1,
        user_id="demo_user",
        delivery_address={"street": "456 Food Ave", "city": "New York", "state": "NY", "zip": "10001"},
        confirm=True
    )
    
    response = await place_order(instacart_request)
    if response.success:
        print(f"Instacart order placed: {response.order.order_id}")
        print(f"   Total: ${response.order.total:.2f}")
    else:
        print(f"Instacart order failed: {response.error}")
    
    print("\nTesting Error Scenarios...")
    
    invalid_request = PlaceOrderRequest("invalid_product", 1, "demo_user", confirm=True)
    response = await place_order(invalid_request)
    print(f"Invalid product: {'FAILED' if not response.success else 'SUCCESS'}")
    
    unconfirmed_request = PlaceOrderRequest("prod_001", 1, "demo_user", confirm=False)
    response = await place_order(unconfirmed_request)
    print(f"Unconfirmed order: {'FAILED' if not response.success else 'SUCCESS'}")
    
    print("\nUser Orders:")
    user_orders = get_user_orders("demo_user")
    for order in user_orders:
        print(f"   {order.order_id}: {order.vendor} - ${order.total:.2f} - {order.status.value}")
    
    print("\nSpending Summary:")
    summary = get_user_spending_summary("demo_user")
    print(f"   Total orders: {summary['total_orders']}")
    print(f"   Total spent: ${summary['total_spent']:.2f}")
    print(f"   Average order: ${summary['average_order_value']:.2f}")


if __name__ == "__main__":
    asyncio.run(demo_automatic_orders())
