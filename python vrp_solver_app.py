import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pulp

class DataParseError(Exception):
    pass


def parse_data_file(filepath):
    """
    Parses the template .txt file into:
        sites: dict  {site_name: {"fixed_cost": float, "capacity": float}}
        clients: dict {client_name: demand}
        costs: dict {site_name: {client_name: cost_per_unit}}
    """
    sites = {}
    clients = {}
    client_order = []
    costs = {}

    section = None

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip().upper()
                continue

            parts = line.split()

            if section == "SITES":
                if len(parts) < 3:
                    raise DataParseError(f"Bad SITES line: '{line}'")
                name, fixed_cost, capacity = parts[0], parts[1], parts[2]
                sites[name] = {
                    "fixed_cost": float(fixed_cost),
                    "capacity": float(capacity),
                }

            elif section == "CLIENTS":
                if len(parts) < 2:
                    raise DataParseError(f"Bad CLIENTS line: '{line}'")
                name, demand = parts[0], parts[1]
                clients[name] = float(demand)
                client_order.append(name)

            elif section == "TRANSPORT_COST":
                if not client_order:
                    raise DataParseError(
                        "TRANSPORT_COST section must come after CLIENTS "
                        "section, or CLIENTS section is empty."
                    )
                site_name = parts[0]
                values = parts[1:]
                if len(values) != len(client_order):
                    raise DataParseError(
                        f"Row for site '{site_name}' has {len(values)} cost "
                        f"values but there are {len(client_order)} clients."
                    )
                costs[site_name] = {
                    client_order[k]: float(values[k]) for k in range(len(values))
                }
            else:
                raise DataParseError(f"Line found outside of a known section: '{line}'")

    if not sites:
        raise DataParseError("No sites found (missing [SITES] section?).")
    if not clients:
        raise DataParseError("No clients found (missing [CLIENTS] section?).")
    if not costs:
        raise DataParseError("No transport costs found (missing [TRANSPORT_COST] section?).")

    missing_cost_sites = set(sites.keys()) - set(costs.keys())
    if missing_cost_sites:
        raise DataParseError(
            f"Missing transport cost rows for sites: {', '.join(missing_cost_sites)}"
        )

    return sites, clients, costs


def solve_facility_location(sites, clients, costs):
    site_names = list(sites.keys())
    client_names = list(clients.keys())

    prob = pulp.LpProblem("Facility_Location_VRP", pulp.LpMinimize)
    y = {i: pulp.LpVariable(f"open_{i}", cat="Binary") for i in site_names}
    x = {
        (i, j): pulp.LpVariable(f"assign_{i}_{j}", cat="Binary")
        for i in site_names
        for j in client_names
    }

    prob += (
        pulp.lpSum(sites[i]["fixed_cost"] * y[i] for i in site_names)
        + pulp.lpSum(
            costs[i][j] * x[(i, j)]
            for i in site_names
            for j in client_names
        )
    )

    for j in client_names:
        prob += pulp.lpSum(x[(i, j)] for i in site_names) == 1, f"assign_once_{j}"


    for i in site_names:
        prob += (
            pulp.lpSum(clients[j] * x[(i, j)] for j in client_names)
            <= sites[i]["capacity"] * y[i]
        ), f"capacity_{i}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]

    result = {
        "status": status,
        "total_cost": pulp.value(prob.objective) if status == "Optimal" else None,
        "opened_sites": [],
        "assignments": {},  # site -> list of (client, demand)
        "site_load": {},
    }

    if status != "Optimal":
        return result

    for i in site_names:
        if y[i].varValue and y[i].varValue > 0.5:
            result["opened_sites"].append(i)
            result["assignments"][i] = []
            result["site_load"][i] = 0.0

    for j in client_names:
        for i in site_names:
            if x[(i, j)].varValue and x[(i, j)].varValue > 0.5:
                result["assignments"].setdefault(i, []).append((j, clients[j]))
                result["site_load"][i] = result["site_load"].get(i, 0.0) + clients[j]
                break

    return result



class VRPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VRP / Facility Location Solver")
        self.root.geometry("700x560")

        self.sites = None
        self.clients = None
        self.costs = None
        self.filepath = None

        top_frame = tk.Frame(root)
        top_frame.pack(pady=12)

        self.import_btn = tk.Button(
            top_frame, text="Import Data (.txt)", width=20,
            command=self.import_data
        )
        self.import_btn.grid(row=0, column=0, padx=8)

        self.solve_btn = tk.Button(
            top_frame, text="Solve", width=20,
            command=self.solve, state=tk.DISABLED
        )
        self.solve_btn.grid(row=0, column=1, padx=8)

        # --- Status label ---
        self.status_label = tk.Label(root, text="No data loaded.", fg="gray")
        self.status_label.pack(pady=(0, 8))

        # --- Output box ---
        self.output_box = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, width=84, height=28, font=("Consolas", 10)
        )
        self.output_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.output_box.insert(
            tk.END,
            "Welcome.\n\n"
            "1. Click 'Import Data (.txt)' and select your filled-in template.\n"
            "2. Once loaded, click 'Solve' to compute the least-cost solution.\n"
        )
        self.output_box.config(state=tk.DISABLED)

    def import_data(self):
        filepath = filedialog.askopenfilename(
            title="Select your data file",
            filetypes=[("Text files", "*.txt")],
        )
        if not filepath:
            return

        try:
            sites, clients, costs = parse_data_file(filepath)
        except DataParseError as e:
            messagebox.showerror("Data Error", f"Could not parse the file:\n\n{e}")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error reading file:\n\n{e}")
            return

        self.sites = sites
        self.clients = clients
        self.costs = costs
        self.filepath = filepath

        self.status_label.config(
            text=f"Loaded: {filepath}  |  {len(sites)} sites, {len(clients)} clients",
            fg="green",
        )
        self.solve_btn.config(state=tk.NORMAL)

        messagebox.showinfo("Success", "Data added successfully")

    def solve(self):
        if not self.sites:
            messagebox.showwarning("No data", "Please import a data file first.")
            return

        self.solve_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Solving...", fg="orange")
        self.root.update_idletasks()

        try:
            result = solve_facility_location(self.sites, self.clients, self.costs)
        except Exception as e:
            messagebox.showerror("Solver Error", f"An error occurred while solving:\n\n{e}")
            self.solve_btn.config(state=tk.NORMAL)
            return

        self.display_result(result)
        self.solve_btn.config(state=tk.NORMAL)

    def display_result(self, result):
        self.output_box.config(state=tk.NORMAL)
        self.output_box.delete("1.0", tk.END)

        if result["status"] != "Optimal":
            self.output_box.insert(
                tk.END, f"No optimal solution found. Solver status: {result['status']}\n"
            )
            self.status_label.config(text=f"Status: {result['status']}", fg="red")
            self.output_box.config(state=tk.DISABLED)
            return

        lines = []
        lines.append("=" * 60)
        lines.append("OPTIMAL SOLUTION")
        lines.append("=" * 60)
        lines.append(f"Total cost: {result['total_cost']:.2f}\n")

        lines.append(f"Opened sites ({len(result['opened_sites'])}):")
        for i in result["opened_sites"]:
            fixed = self.sites[i]["fixed_cost"]
            cap = self.sites[i]["capacity"]
            load = result["site_load"].get(i, 0.0)
            lines.append(
                f"  - {i}: installation cost = {fixed:.2f}, "
                f"capacity = {cap:.2f}, used = {load:.2f} "
                f"({(load/cap*100 if cap else 0):.1f}% utilization)"
            )

        lines.append("")
        lines.append("Client assignments:")
        for i in result["opened_sites"]:
            assigned = result["assignments"].get(i, [])
            lines.append(f"  Site {i} serves {len(assigned)} client(s):")
            for client_name, demand in assigned:
                transport_cost = self.costs[i][client_name]
                lines.append(
                    f"      {client_name}: demand = {demand:.2f}, "
                    f"transport cost = {transport_cost:.2f}"
                )

        self.output_box.insert(tk.END, "\n".join(lines))
        self.output_box.config(state=tk.DISABLED)

        self.status_label.config(
            text=f"Solved. Total cost = {result['total_cost']:.2f}", fg="green"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = VRPApp(root)
    root.mainloop()